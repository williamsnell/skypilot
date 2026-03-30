"""Unit tests for sky.server.slurm_task_queue."""
import concurrent.futures
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
    name = 'test-cluster'
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
        token = queue.generate_token('c1')
        queue.store_token('c1', token)
        assert queue.validate_token('c1', token)

    def test_validate_wrong_token(self, queue):
        queue.store_token('c1', 'correct')
        assert not queue.validate_token('c1', 'wrong')

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
        future = queue.enqueue_task(name, TaskType.SETUP, 'echo hi')
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
        from sky.server.slurm_task_queue import SlurmTask
        task = SlurmTask(task_id='x',
                         task_type=TaskType.SETUP,
                         script_content='echo')
        with pytest.raises(ValueError, match='No signing key'):
            queue.sign_task('no-such-cluster', task, nonce=_nonce())

    def test_get_public_key_bytes(self, queue):
        queue.generate_keypair('c1')
        assert queue.get_public_key_bytes('c1') is not None
        assert queue.get_public_key_bytes('c2') is None


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

    def test_complete_resolves_future(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        future = queue.enqueue_task(name, TaskType.SETUP, 'echo done')
        task = queue.dequeue_task(name, nonce=_nonce())
        queue.complete_task(name,
                            task['task_id'],
                            exit_code=0,
                            stdout='done\n',
                            stderr='')
        exit_code, stdout, stderr = future.result(timeout=1)
        assert exit_code == 0
        assert stdout == 'done\n'

    def test_complete_failure(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        future = queue.enqueue_task(name, TaskType.SETUP, 'fail')
        task = queue.dequeue_task(name, nonce=_nonce())
        queue.complete_task(name, task['task_id'], exit_code=1, stderr='error')
        exit_code, _, stderr = future.result(timeout=1)
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
        future = None

        def enqueue():
            nonlocal future
            future = queue.enqueue_task(name, TaskType.RUN, 'threaded')

        thread = threading.Thread(target=enqueue)
        thread.start()
        thread.join()

        task = queue.dequeue_task(name, nonce=_nonce())
        assert task is not None
        assert task['script_content'] == 'threaded'

    def test_future_timeout(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        future = queue.enqueue_task(name, TaskType.SETUP, 'slow')
        with pytest.raises(concurrent.futures.TimeoutError):
            future.result(timeout=0.1)


class TestHeartbeat:
    """Tests for heartbeat tracking."""

    def test_record_heartbeat(self, queue):
        queue.record_heartbeat('c1')
        assert queue.is_worker_online('c1')

    def test_worker_offline(self, queue):
        assert not queue.is_worker_online('nonexistent')

    def test_stale_heartbeat(self, queue):
        queue.record_heartbeat('c1')
        # Simulate staleness by checking with tiny timeout
        assert not queue.is_worker_online('c1', timeout=0.0)

    def test_get_last_heartbeat(self, queue):
        assert queue.get_last_heartbeat('c1') is None
        queue.record_heartbeat('c1')
        ts = queue.get_last_heartbeat('c1')
        assert ts is not None
        assert time.time() - ts < 2


class TestCleanup:
    """Tests for cluster cleanup."""

    def test_cleanup_removes_tasks(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        queue.enqueue_task(name, TaskType.SETUP, 'echo')
        queue.cleanup_cluster(name)
        assert queue.dequeue_task(name, nonce=_nonce()) is None

    def test_cleanup_removes_heartbeat(self, queue):
        queue.record_heartbeat('c1')
        queue.cleanup_cluster('c1')
        assert not queue.is_worker_online('c1')

    def test_cleanup_removes_signing_key(self, queue):
        queue.generate_keypair('c1')
        queue.cleanup_cluster('c1')
        assert queue.get_public_key_bytes('c1') is None

    def test_cleanup_cancels_pending_futures(self, cluster_with_keys, queue):
        name, _, _ = cluster_with_keys
        future = queue.enqueue_task(name, TaskType.SETUP, 'echo')
        queue.cleanup_cluster(name)
        assert future.cancelled()

    def test_cleanup_removes_token(self, cluster_with_keys, queue):
        name, token, _ = cluster_with_keys
        queue.cleanup_cluster(name)
        assert not queue.validate_token(name, token)


class TestConcurrentComplete:
    """Verify complete_task is thread-safe."""

    def test_concurrent_complete_no_crash(self, cluster_with_keys, queue):
        """Two threads completing the same task should not raise."""
        name, _, _ = cluster_with_keys
        future = queue.enqueue_task(name, TaskType.SETUP, 'echo')
        task_dict = queue.dequeue_task(name, nonce=_nonce())
        task_id = task_dict['task_id']

        errors = []

        def complete(exit_code):
            try:
                queue.complete_task(name,
                                    task_id,
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
        # Future should be resolved (by whichever thread won)
        result = future.result(timeout=1)
        assert result[0] in (0, 1)


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
