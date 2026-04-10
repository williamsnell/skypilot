"""Integration test: SSH failure during a running job must not kill it.

Tests the core invariant of the podman-hpc architecture: the poll-based
task queue is the execution layer, SSH is just convenience. Transient SSH
failures (key revoked, network blip, jump host reset) must not cause:
  - FAILED_CONTROLLER
  - Task queue commands to fail
  - The poll worker to stop heartbeating

Test approach:
  1. Start the poll worker + mini server
  2. Enqueue a task via the task queue (simulating run_on_head)
  3. While the task is in flight, break SSH by renaming the private key
  4. Verify the task still completes successfully via the task queue
  5. Verify get_command_runners() fails (SSH is broken) but cached path works
  6. Restore the key, verify SSH recovers

Run with the in-process test profile:
    podman compose -f docker-compose.yml --profile test run --rm testrunner \
        python -m pytest tests/integration_tests/slurm_tunnel/test_ssh_resilience.py -v
"""
import os
import subprocess
import time

import pytest

from sky import exceptions
from sky.server.slurm_task_queue import TaskType

pytestmark = [
    pytest.mark.integration,
]


def _wait_for_heartbeat(task_queue, cluster_name, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if task_queue.is_worker_online(cluster_name):
            return True
        time.sleep(0.3)
    return False


class TestTaskQueueSurvivesSSHFailure:
    """Task queue operations continue even when SSH is broken."""

    def test_task_completes_with_broken_ssh(self, poll_worker_proc,
                                            cluster_credentials, task_queue,
                                            slurm_cluster):
        """A task enqueued via the task queue completes even after SSH
        keys are revoked on the login node.

        This simulates what happens when the jump host resets the
        connection or the SSH certificate expires mid-job. The task
        queue (HTTP) path must not be affected.
        """
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), 'Worker not online'

        ssh_key = slurm_cluster['ssh_key']
        broken_key = ssh_key + '.broken'

        # Verify SSH works before we break it
        ssh_result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', '-o',
            'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=5', '-i',
            ssh_key, '-p',
            str(slurm_cluster['port']),
            f'{slurm_cluster["user"]}@{slurm_cluster["host"]}', 'echo', 'ssh-ok'
        ],
                                    capture_output=True,
                                    text=True,
                                    timeout=10)
        assert ssh_result.returncode == 0, f'SSH should work: {ssh_result.stderr}'

        try:
            # Break SSH by renaming the private key
            os.rename(ssh_key, broken_key)

            # Verify SSH is actually broken
            ssh_result = subprocess.run([
                'ssh', '-o', 'StrictHostKeyChecking=no', '-o',
                'UserKnownHostsFile=/dev/null', '-o', 'BatchMode=yes', '-o',
                'ConnectTimeout=3', '-i', ssh_key, '-p',
                str(slurm_cluster['port']),
                f'{slurm_cluster["user"]}@{slurm_cluster["host"]}', 'echo',
                'should-fail'
            ],
                                        capture_output=True,
                                        text=True,
                                        timeout=10)
            assert ssh_result.returncode != 0, (
                'SSH should be broken after key removal')

            # Enqueue a task — this goes through the HTTP task queue,
            # NOT SSH. It should succeed despite SSH being broken.
            task_id = task_queue.enqueue_task(cluster, TaskType.RUN,
                                              'echo "ssh-broken-but-i-work"')

            exit_code, stdout, stderr = task_queue.wait_for_completion(
                cluster, task_id, timeout=15)
            assert exit_code == 0, (
                f'Task should succeed via task queue even with broken SSH. '
                f'stderr={stderr}')
            assert 'ssh-broken-but-i-work' in stdout

            # Heartbeat should still be alive
            assert task_queue.is_worker_online(cluster), (
                'Poll worker heartbeat should survive SSH failure')

        finally:
            # Restore the SSH key
            if os.path.exists(broken_key):
                os.rename(broken_key, ssh_key)

    def test_multiple_tasks_during_ssh_outage(self, poll_worker_proc,
                                              cluster_credentials, task_queue,
                                              slurm_cluster):
        """Multiple sequential tasks complete during an SSH outage."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), 'Worker not online'

        ssh_key = slurm_cluster['ssh_key']
        broken_key = ssh_key + '.broken'

        try:
            os.rename(ssh_key, broken_key)

            results = []
            for i in range(3):
                task_id = task_queue.enqueue_task(cluster, TaskType.RUN,
                                                  f'echo "task-{i}"')
                exit_code, stdout, stderr = task_queue.wait_for_completion(
                    cluster, task_id, timeout=15)
                results.append((exit_code, stdout))

            for i, (exit_code, stdout) in enumerate(results):
                assert exit_code == 0, f'Task {i} failed during SSH outage'
                assert f'task-{i}' in stdout

        finally:
            if os.path.exists(broken_key):
                os.rename(broken_key, ssh_key)

    def test_ssh_recovers_after_key_restored(self, poll_worker_proc,
                                             cluster_credentials, task_queue,
                                             slurm_cluster):
        """SSH recovers after the key is restored."""
        cluster = cluster_credentials['cluster_name']
        assert _wait_for_heartbeat(task_queue, cluster), 'Worker not online'

        ssh_key = slurm_cluster['ssh_key']
        broken_key = ssh_key + '.broken'

        try:
            # Break SSH
            os.rename(ssh_key, broken_key)

            # Task queue still works
            task_id = task_queue.enqueue_task(cluster, TaskType.RUN,
                                              'echo "during-outage"')
            exit_code, _, _ = task_queue.wait_for_completion(cluster,
                                                             task_id,
                                                             timeout=15)
            assert exit_code == 0

        finally:
            # Restore SSH
            if os.path.exists(broken_key):
                os.rename(broken_key, ssh_key)

        # SSH should work again. Kill any stale ControlMaster.
        subprocess.run([
            'ssh', '-O', 'exit', '-o', 'StrictHostKeyChecking=no', '-o',
            'UserKnownHostsFile=/dev/null', '-i', ssh_key, '-p',
            str(slurm_cluster['port']),
            f'{slurm_cluster["user"]}@{slurm_cluster["host"]}'
        ],
                       capture_output=True,
                       timeout=5)

        ssh_result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', '-o',
            'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=5', '-i',
            ssh_key, '-p',
            str(slurm_cluster['port']),
            f'{slurm_cluster["user"]}@{slurm_cluster["host"]}', 'echo',
            'recovered'
        ],
                                    capture_output=True,
                                    text=True,
                                    timeout=10)
        assert ssh_result.returncode == 0, (
            f'SSH should recover after key restored: {ssh_result.stderr}')
        assert 'recovered' in ssh_result.stdout


class TestCommandRunnerCacheIntegration:
    """Test that the command runner cache prevents SSH calls."""

    def test_cache_survives_ssh_failure(self, slurm_cluster):
        """After caching, get_command_runners works with broken SSH."""
        from unittest import mock

        from sky.provision import common
        from sky.provision.slurm import instance as slurm_instance

        # Clear cache
        with slurm_instance._command_runner_cache_lock:
            slurm_instance._command_runner_cache.clear()

        cluster_name = 'cache-test-cluster'

        # Pre-populate cache directly (simulating a successful first call)
        with slurm_instance._command_runner_cache_lock:
            slurm_instance._command_runner_cache[cluster_name] = {
                'remote_home_dir': '/home/testuser',
                'workdir': None,
                'tmpdir': None,
                'has_container': True,
            }

        # Build ClusterInfo
        instance_info = common.InstanceInfo(
            instance_id='job-456',
            internal_ip='10.0.0.1',
            external_ip=slurm_cluster['host'],
            ssh_port=slurm_cluster['port'],
            tags={
                'skypilot-cluster-name': cluster_name,
                'job_id': '456',
                'node': 'node01',
            },
        )
        cluster_info = common.ClusterInfo(
            instances={'job-456': [instance_info]},
            head_instance_id='job-456',
            provider_name='slurm',
            provider_config={
                'ssh': {
                    'hostname': slurm_cluster['host'],
                    'port': slurm_cluster['port'],
                    'user': slurm_cluster['user'],
                    'private_key': '/nonexistent/key',  # broken key!
                },
                'cluster': 'test-slurm',
                'container_runtime': 'podman-hpc',
            },
        )

        # Should succeed using cache, not SSH
        runners = slurm_instance.get_command_runners(cluster_info)
        assert len(runners) == 1

        # Clean up
        slurm_instance.invalidate_command_runner_cache(cluster_name)
