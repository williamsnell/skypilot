"""Unit tests for sky.server.slurm_task_queue."""
import multiprocessing
import secrets
import threading
import time

import pytest

from sky.server.slurm_task_queue import _canonical_payload
from sky.server.slurm_task_queue import SlurmTaskQueue
from sky.server.slurm_task_queue import TaskStatus
from sky.server.slurm_task_queue import TaskType


def _nonce() -> str:
    """Generate a random nonce, matching what the poll worker does."""
    return secrets.token_hex(16)


@pytest.fixture
def queue():
    return SlurmTaskQueue()


@pytest.fixture
def cluster_with_keys(queue):
    """A cluster with token and keypair configured."""
    name = f'test-cluster-{secrets.token_hex(4)}'
    token = queue.generate_token(name)
    queue.store_token(name, token)
    pubkey_bytes = queue.generate_keypair(name)
    return name, token, pubkey_bytes


class TestTokenManagement:
    """Tests for token generation, storage, and validation."""

    def test_generate_token_length(self, queue):
        token = queue.generate_token('c1')
        assert len(token) == 64  # 32 bytes hex

    def test_generate_token_unique(self, queue):
        t1 = queue.generate_token('c1')
        t2 = queue.generate_token('c1')
        assert t1 != t2

    def test_store_and_validate(self, queue):
        name = f'store-val-{secrets.token_hex(4)}'
        token = queue.generate_token(name)
        queue.store_token(name, token)
        assert queue.validate_token(name, token)

    def test_validate_wrong_token(self, queue):
        name = f'wrong-tok-{secrets.token_hex(4)}'
        queue.store_token(name, 'correct')
        assert not queue.validate_token(name, 'wrong')

    def test_validate_no_token(self, queue):
        assert not queue.validate_token('nonexistent', 'anything')

    def test_remove_tokens(self, queue):
        queue.store_token('c1', 'tok')
        queue.remove_tokens('c1')
        assert not queue.validate_token('c1', 'tok')


class TestSigningKeys:
    """Tests for Ed25519 keypair generation and task signing."""

    def test_generate_keypair(self, queue):
        pubkey = queue.generate_keypair('c1')
        assert len(pubkey) == 32  # Ed25519 public key is 32 bytes

    def test_generate_keypair_unique(self, queue):
        k1 = queue.generate_keypair('c1')
        k2 = queue.generate_keypair('c2')
        assert k1 != k2

    def test_sign_task(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        queue.enqueue_task(name, TaskType.SETUP, 'echo hi')
        task_dict = queue.dequeue_task(name, nonce=_nonce())
        assert task_dict is not None
        assert 'signature' in task_dict
        assert len(task_dict['signature']) > 0

    def test_sign_and_verify(self, cluster_with_keys, queue):
        name, _, pubkey_bytes = cluster_with_keys
        nonce = _nonce()
        queue.enqueue_task(name, TaskType.SETUP, 'echo hello')
        task_dict = queue.dequeue_task(name, nonce=nonce)

        # Verify with the public key
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey)
        pubkey = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        payload = _canonical_payload(task_dict['task_id'],
                                     task_dict['task_type'],
                                     task_dict['script_content'], nonce)
        sig = bytes.fromhex(task_dict['signature'])
        # Should not raise
        pubkey.verify(sig, payload)

    def test_nonce_in_response(self, cluster_with_keys, queue):
        """The response includes the nonce we sent."""
        name, _, _ = cluster_with_keys
        nonce = _nonce()
        queue.enqueue_task(name, TaskType.SETUP, 'echo hi')
        task_dict = queue.dequeue_task(name, nonce=nonce)
        assert task_dict['nonce'] == nonce

    def test_wrong_nonce_fails_verification(self, cluster_with_keys, queue):
        """Signature verified with a different nonce must fail."""
        name, _, pubkey_bytes = cluster_with_keys
        nonce = _nonce()
        queue.enqueue_task(name, TaskType.SETUP, 'echo hello')
        task_dict = queue.dequeue_task(name, nonce=nonce)

        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey)
        pubkey = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        # Verify with a DIFFERENT nonce — must fail
        wrong_nonce = _nonce()
        payload = _canonical_payload(task_dict['task_id'],
                                     task_dict['task_type'],
                                     task_dict['script_content'], wrong_nonce)
        sig = bytes.fromhex(task_dict['signature'])
        with pytest.raises(InvalidSignature):
            pubkey.verify(sig, payload)

    def test_tampered_content_fails(self, cluster_with_keys, queue):
        name, _, pubkey_bytes = cluster_with_keys
        nonce = _nonce()
        queue.enqueue_task(name, TaskType.SETUP, 'echo hello')
        task_dict = queue.dequeue_task(name, nonce=nonce)

        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey)
        pubkey = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        # Tamper with the content
        payload = _canonical_payload(task_dict['task_id'],
                                     task_dict['task_type'], 'echo EVIL', nonce)
        sig = bytes.fromhex(task_dict['signature'])
        with pytest.raises(InvalidSignature):
            pubkey.verify(sig, payload)

    def test_sign_no_key_raises(self, queue):
        with pytest.raises(ValueError, match='No signing key'):
            queue.sign_task('no-such-cluster',
                            'x',
                            'setup',
                            'echo',
                            nonce=_nonce())

    def test_get_public_key_bytes(self, queue):
        name = f'pubkey-test-{secrets.token_hex(4)}'
        queue.generate_keypair(name)
        assert queue.get_public_key_bytes(name) is not None
        assert queue.get_public_key_bytes(
            f'no-such-{secrets.token_hex(4)}') is None


class TestTaskLifecycle:
    """Tests for enqueue, dequeue, and completion."""

    def test_enqueue_and_dequeue(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        queue.enqueue_task(name, TaskType.SETUP, 'echo hi')
        task = queue.dequeue_task(name, nonce=_nonce())
        assert task is not None
        assert task['task_type'] == 'setup'
        assert task['script_content'] == 'echo hi'

    def test_dequeue_empty(self, queue):
        assert queue.dequeue_task('empty', nonce=_nonce()) is None

    def test_fifo_order(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        queue.enqueue_task(name, TaskType.SETUP, 'first')
        queue.enqueue_task(name, TaskType.RUN, 'second')
        t1 = queue.dequeue_task(name, nonce=_nonce())
        t2 = queue.dequeue_task(name, nonce=_nonce())
        assert t1['script_content'] == 'first'
        assert t2['script_content'] == 'second'

    def test_dequeue_skips_running(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        queue.enqueue_task(name, TaskType.SETUP, 'first')
        queue.enqueue_task(name, TaskType.RUN, 'second')
        queue.dequeue_task(name, nonce=_nonce())  # marks first as RUNNING
        t2 = queue.dequeue_task(name, nonce=_nonce())
        assert t2['script_content'] == 'second'

    def test_complete_and_wait(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        task_id = queue.enqueue_task(name, TaskType.SETUP, 'echo done')
        task = queue.dequeue_task(name, nonce=_nonce())

        # Complete in a background thread.
        def complete():
            time.sleep(0.2)
            queue.complete_task(name,
                                task['task_id'],
                                exit_code=0,
                                stdout='done\n',
                                stderr='')

        t = threading.Thread(target=complete)
        t.start()

        exit_code, stdout, stderr = queue.wait_for_completion(name,
                                                              task_id,
                                                              timeout=5)
        t.join()
        assert exit_code == 0
        assert stdout == 'done\n'

    def test_complete_failure(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        task_id = queue.enqueue_task(name, TaskType.SETUP, 'fail')
        task = queue.dequeue_task(name, nonce=_nonce())

        queue.complete_task(name, task['task_id'], exit_code=1, stderr='error')
        exit_code, _, stderr = queue.wait_for_completion(name,
                                                         task_id,
                                                         timeout=5)
        assert exit_code == 1
        assert stderr == 'error'

    def test_complete_unknown_task(self, queue):
        assert not queue.complete_task('c1', 'nonexistent', 0)

    def test_env_vars_passed(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        queue.enqueue_task(name,
                           TaskType.SETUP,
                           'echo $FOO',
                           env_vars={'FOO': 'bar'})
        task = queue.dequeue_task(name, nonce=_nonce())
        assert task['env_vars'] == {'FOO': 'bar'}

    def test_cross_thread_enqueue_dequeue(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        task_id = None

        def enqueue():
            nonlocal task_id
            task_id = queue.enqueue_task(name, TaskType.RUN, 'threaded')

        thread = threading.Thread(target=enqueue)
        thread.start()
        thread.join()

        task = queue.dequeue_task(name, nonce=_nonce())
        assert task is not None
        assert task['script_content'] == 'threaded'

    def test_wait_timeout(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        task_id = queue.enqueue_task(name, TaskType.SETUP, 'slow')
        with pytest.raises(TimeoutError):
            queue.wait_for_completion(name, task_id, timeout=0.2)


class TestHeartbeat:
    """Tests for heartbeat tracking."""

    def _unique_name(self):
        return f'hb-test-{secrets.token_hex(4)}'

    def test_record_heartbeat(self, queue):
        name = self._unique_name()
        queue.record_heartbeat(name)
        assert queue.is_worker_online(name)

    def test_worker_offline(self, queue):
        assert not queue.is_worker_online(f'nonexistent-{secrets.token_hex(4)}')

    def test_stale_heartbeat(self, queue):
        name = self._unique_name()
        queue.record_heartbeat(name)
        # Simulate staleness by checking with tiny timeout
        assert not queue.is_worker_online(name, timeout=0.0)

    def test_get_last_heartbeat(self, queue):
        name = self._unique_name()
        assert queue.get_last_heartbeat(name) is None
        queue.record_heartbeat(name)
        ts = queue.get_last_heartbeat(name)
        assert ts is not None
        assert time.time() - ts < 2


def _heartbeat_writer(cluster_name):
    """Write a heartbeat in a subprocess (simulates uvicorn worker)."""
    from sky.server.slurm_task_queue import SlurmTaskQueue
    queue = SlurmTaskQueue()
    queue.store_token(cluster_name, 'test-token')
    queue.record_heartbeat(cluster_name)


def _heartbeat_reader(cluster_name, result_dict):
    """Read heartbeat in a subprocess (simulates request executor)."""
    from sky.server.slurm_task_queue import SlurmTaskQueue
    queue = SlurmTaskQueue()
    result_dict['is_online'] = queue.is_worker_online(cluster_name)
    result_dict['last_hb'] = queue.get_last_heartbeat(cluster_name)


class TestHeartbeatCrossProcess:
    """Heartbeat must be visible across separate processes (kv_cache DB)."""

    def test_cross_process_heartbeat_visible(self):
        cluster = f'xproc-test-{secrets.token_hex(4)}'

        # Write heartbeat in one process.
        p1 = multiprocessing.Process(target=_heartbeat_writer, args=(cluster,))
        p1.start()
        p1.join()
        assert p1.exitcode == 0

        # Read heartbeat in a different process.
        manager = multiprocessing.Manager()
        result = manager.dict()
        p2 = multiprocessing.Process(target=_heartbeat_reader,
                                     args=(cluster, result))
        p2.start()
        p2.join()
        assert p2.exitcode == 0

        assert result['is_online'], (
            'Heartbeat written in process A must be visible in process B '
            'via kv_cache DB')
        assert result['last_hb'] is not None
        assert time.time() - result['last_hb'] < 5


def _task_enqueuer(cluster_name, result_dict):
    """Enqueue a task in a subprocess (simulates request executor)."""
    from sky.server.slurm_task_queue import SlurmTaskQueue
    from sky.server.slurm_task_queue import TaskType
    queue = SlurmTaskQueue()
    task_id = queue.enqueue_task(cluster_name, TaskType.SETUP, 'echo cross')
    result_dict['task_id'] = task_id


def _task_dequeuer(cluster_name, result_dict):
    """Dequeue a task in a subprocess (simulates FastAPI server)."""
    import secrets as _secrets

    from sky.server.slurm_task_queue import SlurmTaskQueue
    queue = SlurmTaskQueue()
    task = queue.dequeue_task(cluster_name, nonce=_secrets.token_hex(16))
    result_dict['task'] = task


class TestTaskCrossProcess:
    """Tasks enqueued in one process must be visible in another."""

    def test_cross_process_task_visible(self):
        cluster = f'xproc-task-{secrets.token_hex(4)}'

        # Set up signing key (needed for dequeue to sign).
        queue = SlurmTaskQueue()
        queue.generate_keypair(cluster)

        # Enqueue in one process.
        manager = multiprocessing.Manager()
        enqueue_result = manager.dict()
        p1 = multiprocessing.Process(target=_task_enqueuer,
                                     args=(cluster, enqueue_result))
        p1.start()
        p1.join()
        assert p1.exitcode == 0
        assert 'task_id' in enqueue_result

        # Dequeue in a different process.
        dequeue_result = manager.dict()
        p2 = multiprocessing.Process(target=_task_dequeuer,
                                     args=(cluster, dequeue_result))
        p2.start()
        p2.join()
        assert p2.exitcode == 0

        task = dequeue_result.get('task')
        assert task is not None, (
            'Task enqueued in process A must be visible in process B '
            'via kv_cache DB')
        assert task['script_content'] == 'echo cross'


class TestCleanup:
    """Tests for cluster cleanup."""

    def test_cleanup_removes_tasks(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        queue.enqueue_task(name, TaskType.SETUP, 'echo')
        queue.cleanup_cluster(name)
        assert queue.dequeue_task(name, nonce=_nonce()) is None

    def test_cleanup_removes_heartbeat(self, queue):
        name = f'cleanup-hb-{secrets.token_hex(4)}'
        queue.record_heartbeat(name)
        queue.cleanup_cluster(name)
        assert not queue.is_worker_online(name)

    def test_cleanup_removes_signing_key(self, queue):
        queue.generate_keypair('c1')
        queue.cleanup_cluster('c1')
        assert queue.get_public_key_bytes('c1') is None

    def test_cleanup_completes_pending_task(self, cluster_with_keys, queue):
        """Cleanup writes a cancellation result for pending tasks."""
        name, _, _ = cluster_with_keys
        task_id = queue.enqueue_task(name, TaskType.SETUP, 'echo')
        queue.cleanup_cluster(name)
        # The result should be available (cancellation).
        exit_code, _, stderr = queue.wait_for_completion(name,
                                                         task_id,
                                                         timeout=1)
        assert exit_code == -1
        assert 'cancelled' in stderr.lower()

    def test_cleanup_removes_token(self, cluster_with_keys, queue):
        name, token, _ = cluster_with_keys
        queue.cleanup_cluster(name)
        assert not queue.validate_token(name, token)


class TestConcurrentComplete:
    """Verify complete_task is thread-safe."""

    def test_concurrent_complete_no_crash(self, cluster_with_keys, queue):
        """Two threads completing the same task should not raise."""
        name, _, _ = cluster_with_keys
        task_id = queue.enqueue_task(name, TaskType.SETUP, 'echo')
        task_dict = queue.dequeue_task(name, nonce=_nonce())
        tid = task_dict['task_id']

        errors = []

        def complete(exit_code):
            try:
                queue.complete_task(name,
                                    tid,
                                    exit_code=exit_code,
                                    stdout=f'out-{exit_code}')
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=complete, args=(0,))
        t2 = threading.Thread(target=complete, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f'Concurrent complete raised: {errors}'
        # Result should be available.
        exit_code, stdout, _ = queue.wait_for_completion(name,
                                                         task_id,
                                                         timeout=1)
        assert exit_code in (0, 1)


class TestTokenAuthRejection:
    """Tests for token validation edge cases."""

    def test_empty_token_rejected(self, queue):
        queue.store_token('c1', 'real-token')
        assert not queue.validate_token('c1', '')

    def test_none_cluster_rejected(self, queue):
        assert not queue.validate_token('', 'any-token')

    def test_timing_safe_comparison(self, queue):
        """Validate uses constant-time comparison (secrets.compare_digest)."""
        queue.store_token('c1', 'correct')
        # These should all fail without timing leaks
        assert not queue.validate_token('c1', 'correc')
        assert not queue.validate_token('c1', 'correctx')
        assert not queue.validate_token('c1', 'CORRECT')


class TestKeyNameValidation:
    """Tests for skypilot_* key name validation."""

    def test_valid_key_name(self):
        from sky.provision.slurm.utils import validate_identity_file_for_remote

        # Should not raise
        validate_identity_file_for_remote('/home/user/.ssh/skypilot_slurm')
        validate_identity_file_for_remote('/tmp/skypilot_test')

    def test_rejects_standard_key_names(self):
        from sky.provision.slurm.utils import validate_identity_file_for_remote
        with pytest.raises(ValueError, match='skypilot_'):
            validate_identity_file_for_remote('/home/user/.ssh/id_rsa')
        with pytest.raises(ValueError, match='skypilot_'):
            validate_identity_file_for_remote('/home/user/.ssh/id_ed25519')
        with pytest.raises(ValueError, match='skypilot_'):
            validate_identity_file_for_remote('/home/user/.ssh/my_key')

    def test_rejects_key_with_prefix_in_directory(self):
        """The prefix must be on the filename, not a parent directory."""
        from sky.provision.slurm.utils import validate_identity_file_for_remote
        with pytest.raises(ValueError, match='skypilot_'):
            validate_identity_file_for_remote('/home/skypilot_user/.ssh/id_rsa')
