"""Integration tests for the poll-based Slurm task queue.

Run inside the testrunner container:
    podman compose -f tests/integration_tests/slurm_tunnel/docker-compose.yml \
        up -d --build
    podman compose -f tests/integration_tests/slurm_tunnel/docker-compose.yml \
        run --rm --service-ports testrunner \
        python -m pytest tests/integration_tests/slurm_tunnel/ -v

Or on the host (requires Slurm cluster running via compose):
    python -m pytest tests/integration_tests/slurm_tunnel/ -v
"""
import subprocess
import time

import pytest

from sky.server.slurm_task_queue import TaskType

pytestmark = [
    pytest.mark.integration,
]


def _wait_for_heartbeat(task_queue, cluster_name, timeout=15):
    """Wait for the poll worker to send its first heartbeat."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if task_queue.is_worker_online(cluster_name):
            return True
        time.sleep(0.3)
    return False


class TestSlurmClusterConnectivity:
    """Verify the Slurm cluster is reachable directly (sanity check)."""

    def test_ssh_works(self, slurm_cluster):
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', '-o',
            'UserKnownHostsFile=/dev/null', '-i', slurm_cluster['ssh_key'],
            '-p',
            str(slurm_cluster['port']),
            f'{slurm_cluster["user"]}@{slurm_cluster["host"]}', 'echo', 'ssh-ok'
        ],
                                capture_output=True,
                                text=True,
                                timeout=10)
        assert result.returncode == 0, f'SSH failed: {result.stderr}'
        assert 'ssh-ok' in result.stdout

    def test_sinfo(self, slurm_cluster):
        """Verify slurmctld responds to sinfo.

        Note: with multi-network docker-compose, workers may not register
        due to hostname resolution across networks. We only check that
        sinfo runs successfully and partitions exist.
        """
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', '-o',
            'UserKnownHostsFile=/dev/null', '-i', slurm_cluster['ssh_key'],
            '-p',
            str(slurm_cluster['port']),
            f'{slurm_cluster["user"]}@{slurm_cluster["host"]}', 'sinfo',
            '--noheader'
        ],
                                capture_output=True,
                                text=True,
                                timeout=10)
        assert result.returncode == 0, f'sinfo failed: {result.stderr}'
        assert 'up' in result.stdout, f'No partitions up: {result.stdout}'


class TestPollWorkerHeartbeat:
    """Test that the poll worker connects and sends heartbeats."""

    def test_worker_heartbeats(self, poll_worker_proc, cluster_credentials,
                               task_queue):
        cluster = cluster_credentials['cluster_name']
        online = _wait_for_heartbeat(task_queue, cluster)
        if not online:
            poll_worker_proc.terminate()
            _, stderr = poll_worker_proc.communicate(timeout=3)
            pytest.fail(f'Poll worker did not heartbeat. '
                        f'stderr: {stderr.decode(errors="replace")}')
        assert task_queue.is_worker_online(cluster)


class TestTaskExecution:
    """Test that the poll worker picks up and executes tasks."""

    def test_setup_task(self, poll_worker_proc, cluster_credentials,
                        task_queue):
        """Enqueue a setup task and verify the worker runs it."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        future = task_queue.enqueue_task(cluster, TaskType.SETUP,
                                         'echo "setup-output-42"')

        exit_code, stdout, stderr = future.result(timeout=15)
        assert exit_code == 0, f'Setup failed: stderr={stderr}'
        assert 'setup-output-42' in stdout

    def test_run_task(self, poll_worker_proc, cluster_credentials, task_queue):
        """Enqueue a run task and verify the worker runs it."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        future = task_queue.enqueue_task(cluster, TaskType.RUN,
                                         'echo "run-result-99"')

        exit_code, stdout, stderr = future.result(timeout=15)
        assert exit_code == 0, f'Run failed: stderr={stderr}'
        assert 'run-result-99' in stdout

    def test_failing_task(self, poll_worker_proc, cluster_credentials,
                          task_queue):
        """Enqueue a failing task and verify error is reported."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        future = task_queue.enqueue_task(cluster, TaskType.SETUP, 'exit 42')

        exit_code, stdout, stderr = future.result(timeout=15)
        assert exit_code == 42

    def test_env_vars(self, poll_worker_proc, cluster_credentials, task_queue):
        """Verify environment variables are passed to the task."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        future = task_queue.enqueue_task(cluster,
                                         TaskType.SETUP,
                                         'echo "VAL=$MY_TEST_VAR"',
                                         env_vars={'MY_TEST_VAR': 'hello123'})

        exit_code, stdout, stderr = future.result(timeout=15)
        assert exit_code == 0, f'Failed: stderr={stderr}'
        assert 'VAL=hello123' in stdout

    def test_sequential_tasks(self, poll_worker_proc, cluster_credentials,
                              task_queue):
        """Enqueue multiple tasks and verify FIFO execution."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        f1 = task_queue.enqueue_task(cluster, TaskType.SETUP, 'echo "first"')
        f2 = task_queue.enqueue_task(cluster, TaskType.SETUP, 'echo "second"')
        f3 = task_queue.enqueue_task(cluster, TaskType.SETUP, 'echo "third"')

        r1 = f1.result(timeout=15)
        r2 = f2.result(timeout=15)
        r3 = f3.result(timeout=15)

        assert r1[0] == 0 and 'first' in r1[1]
        assert r2[0] == 0 and 'second' in r2[1]
        assert r3[0] == 0 and 'third' in r3[1]


class TestSignatureRejection:
    """Test that the poll worker rejects tasks with invalid signatures."""

    def test_tampered_signature_rejected(self, poll_worker_proc,
                                         cluster_credentials, task_queue):
        """Tamper with the signing key so signatures don't verify."""
        from sky.server.slurm_task_queue import SlurmTask
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        # Enqueue a task, then tamper with its signature before
        # the worker picks it up. We do this by replacing the
        # cluster's signing key with a different one.
        future = task_queue.enqueue_task(cluster, TaskType.SETUP,
                                         'echo "SHOULD NOT RUN"')

        # Replace the signing key — the already-queued task was signed
        # with the old key, but dequeue will re-sign with the new one.
        # Actually, dequeue signs at dequeue time. So we need to
        # tamper differently: swap the key BEFORE dequeue.
        #
        # The cleanest way: generate a new keypair for this cluster,
        # which replaces the signing key. The poll worker still has
        # the OLD public key, so verification will fail.
        task_queue.generate_keypair(cluster)

        exit_code, stdout, stderr = future.result(timeout=15)
        # Poll worker should reject and report failure
        assert exit_code == -1
        assert 'Signature verification failed' in stderr


class TestTokenRejection:
    """Test that invalid tokens are rejected by the mini server."""

    def test_poll_with_bad_token(self, mini_server):
        """Polling with an invalid token returns 401."""
        import urllib.error
        import urllib.request
        url = f'http://127.0.0.1:{mini_server}/slurm/tasks/fake-cluster'
        req = urllib.request.Request(url,
                                     headers={
                                         'X-Slurm-Token': 'wrong',
                                         'X-Slurm-Nonce': 'dummy',
                                     })
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 401

    def test_heartbeat_with_bad_token(self, mini_server):
        """Heartbeat with an invalid token returns 401."""
        import json
        import urllib.error
        import urllib.request
        url = (f'http://127.0.0.1:{mini_server}'
               f'/slurm/tasks/fake-cluster/heartbeat')
        req = urllib.request.Request(url,
                                     data=json.dumps({}).encode(),
                                     headers={
                                         'X-Slurm-Token': 'wrong',
                                         'Content-Type': 'application/json'
                                     },
                                     method='POST')
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 401
