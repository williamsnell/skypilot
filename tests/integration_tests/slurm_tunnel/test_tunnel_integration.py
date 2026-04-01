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
import os
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

        task_id = task_queue.enqueue_task(cluster, TaskType.SETUP,
                                          'echo "setup-output-42"')

        exit_code, stdout, stderr = task_queue.wait_for_completion(cluster,
                                                                   task_id,
                                                                   timeout=15)
        assert exit_code == 0, f'Setup failed: stderr={stderr}'
        assert 'setup-output-42' in stdout

    def test_run_task(self, poll_worker_proc, cluster_credentials, task_queue):
        """Enqueue a run task and verify the worker runs it."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        task_id = task_queue.enqueue_task(cluster, TaskType.RUN,
                                          'echo "run-result-99"')

        exit_code, stdout, stderr = task_queue.wait_for_completion(cluster,
                                                                   task_id,
                                                                   timeout=15)
        assert exit_code == 0, f'Run failed: stderr={stderr}'
        assert 'run-result-99' in stdout

    def test_failing_task(self, poll_worker_proc, cluster_credentials,
                          task_queue):
        """Enqueue a failing task and verify error is reported."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        task_id = task_queue.enqueue_task(cluster, TaskType.SETUP, 'exit 42')

        exit_code, stdout, stderr = task_queue.wait_for_completion(cluster,
                                                                   task_id,
                                                                   timeout=15)
        assert exit_code == 42

    def test_env_vars(self, poll_worker_proc, cluster_credentials, task_queue):
        """Verify environment variables are passed to the task."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        task_id = task_queue.enqueue_task(cluster,
                                          TaskType.SETUP,
                                          'echo "VAL=$MY_TEST_VAR"',
                                          env_vars={'MY_TEST_VAR': 'hello123'})

        exit_code, stdout, stderr = task_queue.wait_for_completion(cluster,
                                                                   task_id,
                                                                   timeout=15)
        assert exit_code == 0, f'Failed: stderr={stderr}'
        assert 'VAL=hello123' in stdout

    def test_sequential_tasks(self, poll_worker_proc, cluster_credentials,
                              task_queue):
        """Enqueue multiple tasks and verify FIFO execution."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        t1 = task_queue.enqueue_task(cluster, TaskType.SETUP, 'echo "first"')
        t2 = task_queue.enqueue_task(cluster, TaskType.SETUP, 'echo "second"')
        t3 = task_queue.enqueue_task(cluster, TaskType.SETUP, 'echo "third"')

        r1 = task_queue.wait_for_completion(cluster, t1, timeout=15)
        r2 = task_queue.wait_for_completion(cluster, t2, timeout=15)
        r3 = task_queue.wait_for_completion(cluster, t3, timeout=15)

        assert r1[0] == 0 and 'first' in r1[1]
        assert r2[0] == 0 and 'second' in r2[1]
        assert r3[0] == 0 and 'third' in r3[1]


class TestSignatureRejection:
    """Test that the poll worker rejects tasks with invalid signatures."""

    def test_tampered_signature_rejected(self, poll_worker_proc,
                                         cluster_credentials, task_queue):
        """Tamper with the signing key so signatures don't verify."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), \
            'Worker not online'

        # Enqueue a task, then tamper with its signature before
        # the worker picks it up. We do this by replacing the
        # cluster's signing key with a different one.
        task_id = task_queue.enqueue_task(cluster, TaskType.SETUP,
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

        exit_code, stdout, stderr = task_queue.wait_for_completion(cluster,
                                                                   task_id,
                                                                   timeout=15)
        # Poll worker should reject and report failure
        assert exit_code == -1
        assert 'Signature verification failed' in stderr


class TestTerminateInstances:
    """Test that terminate_instances cancels Slurm jobs via scancel."""

    def test_running_job_is_cancelled(self, slurm_cluster):
        """Submit a sleep job, call terminate_instances, verify it leaves
        squeue entirely (not just receives a signal).

        Regression test: scancel --signal TERM only delivers SIGTERM without
        actually cancelling the Slurm job. The job stays in RUNNING state
        indefinitely if processes survive the signal. A plain scancel (no
        --signal) is needed to tell Slurm to cancel the job.
        """
        cluster_name = f'terminate-test-{int(time.time())}'
        ssh_cmd = [
            'ssh',
            '-o',
            'StrictHostKeyChecking=no',
            '-o',
            'UserKnownHostsFile=/dev/null',
            '-i',
            slurm_cluster['ssh_key'],
            '-p',
            str(slurm_cluster['port']),
            f'{slurm_cluster["user"]}@{slurm_cluster["host"]}',
        ]

        # Submit a job that ignores SIGTERM and loops. This is the
        # realistic case: the sbatch script has long-lived children
        # (Dropbear, poll worker) that survive SIGTERM.
        # The while-loop restarts sleep after each SIGTERM kills it.
        wrap_script = "trap '' TERM; while true; do sleep 10; done"
        sbatch_cmd = (f'sbatch --job-name {cluster_name} '
                      f'--wrap "{wrap_script}"')
        result = subprocess.run(ssh_cmd + [sbatch_cmd],
                                capture_output=True,
                                text=True,
                                timeout=10)
        assert result.returncode == 0, f'sbatch failed: {result.stderr}'
        assert 'Submitted batch job' in result.stdout

        # Verify job is running/pending in squeue.
        squeue_cmd = (f'squeue --name {cluster_name} '
                      f'--noheader --format="%i %j %T"')
        result = subprocess.run(ssh_cmd + [squeue_cmd],
                                capture_output=True,
                                text=True,
                                timeout=10)
        assert result.stdout.strip(), (
            f'Job not found after sbatch: {result.stdout}')

        # Call terminate_instances with SSH config.
        from sky.provision.slurm.instance import terminate_instances
        provider_config = {
            'ssh': {
                'hostname': slurm_cluster['host'],
                'port': slurm_cluster['port'],
                'user': slurm_cluster['user'],
                'private_key': slurm_cluster['ssh_key'],
            }
        }
        terminate_instances(cluster_name, provider_config=provider_config)

        # Poll squeue until the job disappears. A plain scancel causes
        # Slurm to cancel the job (SIGTERM -> KillWait -> SIGKILL), so it
        # should leave squeue within a few seconds. With the buggy
        # --signal TERM, the job stays RUNNING indefinitely.
        deadline = time.time() + 15
        last_squeue = ''
        while time.time() < deadline:
            time.sleep(1)
            result = subprocess.run(ssh_cmd + [squeue_cmd],
                                    capture_output=True,
                                    text=True,
                                    timeout=10)
            last_squeue = result.stdout.strip()
            if not last_squeue:
                break
        assert not last_squeue, (
            f'Job still in squeue 15s after terminate_instances. '
            f'scancel must actually cancel the job, not just send a signal. '
            f'squeue output: {last_squeue}')

    def test_already_terminated_job_is_noop(self, slurm_cluster):
        """terminate_instances on a nonexistent job should not raise."""
        from sky.provision.slurm.instance import terminate_instances
        provider_config = {
            'ssh': {
                'hostname': slurm_cluster['host'],
                'port': slurm_cluster['port'],
                'user': slurm_cluster['user'],
                'private_key': slurm_cluster['ssh_key'],
            }
        }
        # Should not raise — job doesn't exist.
        terminate_instances('nonexistent-cluster-xyz',
                            provider_config=provider_config)


class TestSSHCertificateExpiry:
    """Test that expired SSH certificates are caught before SSH."""

    def test_terminate_with_expired_cert_raises(self, slurm_cluster):
        """terminate_instances with an expired certificate must raise
        SSHCertificateExpiredError, not a cryptic SSH error.

        This is the code path hit by sky down when the user's SSH
        certificate has expired since launch.
        """
        import tempfile

        from sky import exceptions
        from sky.provision.slurm.instance import terminate_instances

        tmpdir = tempfile.mkdtemp()
        ca_key = os.path.join(tmpdir, 'ca')
        user_key = os.path.join(tmpdir, 'user')

        # Generate an expired certificate.
        subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', ca_key, '-N', ''],
                       check=True,
                       capture_output=True)
        subprocess.run(
            ['ssh-keygen', '-t', 'ed25519', '-f', user_key, '-N', ''],
            check=True,
            capture_output=True)
        subprocess.run([
            'ssh-keygen', '-s', ca_key, '-I', 'test-expired', '-V', '-1d:-1s',
            user_key + '.pub'
        ],
                       check=True,
                       capture_output=True)

        provider_config = {
            'ssh': {
                'hostname': slurm_cluster['host'],
                'port': slurm_cluster['port'],
                'user': slurm_cluster['user'],
                'private_key': user_key,
                'certificate_file': user_key + '-cert.pub',
            }
        }

        with pytest.raises(exceptions.SSHCertificateExpiredError):
            terminate_instances('some-cluster', provider_config=provider_config)


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
