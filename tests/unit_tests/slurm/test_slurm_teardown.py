"""Regression tests for poll worker liveness and teardown behavior.

Test 1: The poll worker must continue sending heartbeats while
executing a long-running task. Without this, is_worker_online()
returns False during task execution, making it impossible to
distinguish a busy worker from a dead one.

Test 2: wait_for_completion must fail fast when the poll worker is
genuinely offline (never heartbeated), rather than blocking for the
full timeout. This prevents sky down, sky exec, etc. from hanging
when the container or poll worker has died.

Test 3: teardown_no_lock must complete quickly for Slurm clusters
even when the poll worker is offline (the normal case during
teardown, since the container is being killed).
"""
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from unittest import mock

from sky.backends import cloud_vm_ray_backend
from sky.clouds import cloud as cloud_mod
from sky.server.slurm_task_queue import get_task_queue
from sky.server.slurm_task_queue import SlurmTaskQueue
from sky.server.slurm_task_queue import TaskType
from sky.utils import status_lib


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_heartbeat_continues_during_task_execution():
    """The poll worker must send heartbeats while a task is running.

    If heartbeats stop during execution, is_worker_online() returns
    False for any task longer than HEARTBEAT_TIMEOUT_SECONDS (60s),
    making it impossible to tell if the worker is busy vs dead.
    """
    queue = SlurmTaskQueue()
    cluster = f'hb-exec-{secrets.token_hex(4)}'
    port = _free_port()

    # Set up auth.
    token = queue.generate_token(cluster)
    queue.store_token(cluster, token)
    pubkey_bytes = queue.generate_keypair(cluster)

    # Write pubkey to temp file.
    pubkey_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pub')
    pubkey_file.write(pubkey_bytes)
    pubkey_file.close()

    # Start mini FastAPI server in a subprocess.
    # Server reads token + signing key from kv_cache (shared DB).
    # Do NOT regenerate keypair — that would invalidate the pubkey file.
    server_proc = subprocess.Popen(
        [
            sys.executable, '-c', f"""
import uvicorn
import fastapi
from sky.server.slurm_task_queue import create_poll_router

app = fastapi.FastAPI()
app.include_router(create_poll_router())

@app.get('/api/health')
async def health():
    return {{'status': 'ok'}}

uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')
"""
        ],
        env={
            **os.environ, 'PYTHONPATH': '.'
        },
    )

    # Wait for server to accept connections.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            import urllib.request  # pylint: disable=import-outside-toplevel
            urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',
                                   timeout=1)
            break
        except Exception:  # pylint: disable=broad-except
            time.sleep(0.3)

    # Start poll worker subprocess.
    worker_proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'sky.provision.slurm.poll_worker',
            '--api-server-url',
            f'http://127.0.0.1:{port}',
            '--cluster-name',
            cluster,
            '--server-pubkey-file',
            pubkey_file.name,
        ],
        env={
            **os.environ,
            'SKYPILOT_POLL_TOKEN': token,
            'SKYPILOT_POLL_ALLOW_HTTP': '1',
            'SKYPILOT_POLL_HEARTBEAT_INTERVAL': '0.1',
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for worker to send first heartbeat.
        deadline = time.time() + 10
        while time.time() < deadline:
            if queue.is_worker_online(cluster):
                break
            time.sleep(0.2)
        assert queue.is_worker_online(cluster), 'Worker never came online'

        # Enqueue a task that runs long enough for several heartbeats
        # (at 0.1s interval, 1s is plenty).
        task_id = queue.enqueue_task(cluster, TaskType.RUN, 'sleep 1')

        # Wait for the task to be picked up, then sample heartbeats.
        time.sleep(0.3)

        heartbeats_during_task = []
        for _ in range(5):
            ts = queue.get_last_heartbeat(cluster)
            heartbeats_during_task.append(ts)
            time.sleep(0.2)

        # Wait for task completion.
        exit_code, _, _ = queue.wait_for_completion(cluster,
                                                    task_id,
                                                    timeout=10)
        assert exit_code == 0, 'Sleep task failed'

        # Check that heartbeats continued during execution.
        # With the fix, we should see timestamps updating.
        # Without the fix, all timestamps will be the same (stale).
        unique_timestamps = set(heartbeats_during_task)
        assert len(unique_timestamps) > 1, (
            f'Heartbeat did not update during task execution. '
            f'All timestamps: {heartbeats_during_task}. '
            f'The poll worker must send heartbeats in a background '
            f'thread so is_worker_online() remains accurate during '
            f'long-running tasks.')
    finally:
        worker_proc.terminate()
        worker_proc.wait(timeout=5)
        server_proc.terminate()
        server_proc.wait(timeout=5)
        os.unlink(pubkey_file.name)


def test_wait_for_completion_fails_fast_when_worker_offline():
    """wait_for_completion must not block for the full timeout when
    the poll worker has never heartbeated.

    This covers all callers of run_on_head on podman-hpc clusters:
    teardown, job submission, sky exec, sky queue, etc. Without this
    check, any of these operations hang for up to 3600s when the
    container or poll worker is dead.
    """
    queue = SlurmTaskQueue()
    cluster = f'offline-test-{secrets.token_hex(4)}'
    queue.generate_keypair(cluster)

    # No heartbeat — simulates dead poll worker.
    assert not queue.is_worker_online(cluster)

    task_id = queue.enqueue_task(cluster, TaskType.RUN, 'echo hello')

    start = time.time()
    try:
        queue.wait_for_completion(cluster, task_id, timeout=30)
        assert False, 'Should have raised when worker is offline'
    except (TimeoutError, RuntimeError):
        pass
    elapsed = time.time() - start

    # Should detect offline worker and fail in seconds, not minutes.
    assert elapsed < 10, (
        f'wait_for_completion took {elapsed:.1f}s with no heartbeat. '
        f'Must check is_worker_online() and fail fast when the poll '
        f'worker is dead, not block for the full timeout.')


def test_teardown_completes_quickly_for_slurm():
    """sky down on a Slurm cluster must complete quickly even when
    the poll worker is offline.

    This exercises the real teardown_no_lock code path with a Slurm
    handle. Only infrastructure calls (SSH, provisioner) are mocked.
    The task queue run_on_head path runs for real — if it hangs,
    this test will fail on the time assertion.
    """
    backend = cloud_vm_ray_backend.CloudVmRayBackend()

    # Build a mock handle that looks like a Slurm+podman-hpc cluster.
    mock_cloud = mock.MagicMock(spec=cloud_mod.Cloud)
    mock_cloud.uses_ray.return_value = False
    mock_cloud.PROVISIONER_VERSION = cloud_mod.ProvisionerVersion.SKYPILOT
    mock_cloud.__repr__ = mock.Mock(return_value='Slurm')

    class FakeResources:
        pass

    mock_resources = FakeResources()
    mock_resources.cloud = mock_cloud
    mock_resources.assert_launchable = lambda: mock_resources

    cluster_name = f'teardown-test-{secrets.token_hex(4)}'
    mock_handle = mock.MagicMock()
    mock_handle.launched_resources = mock_resources
    mock_handle.cluster_name = cluster_name
    mock_handle.cluster_name_on_cloud = f'{cluster_name}-abc123'
    mock_handle.cluster_yaml = '/tmp/fake.yaml'

    mock_teardown = mock.patch('sky.provision.provisioner.teardown_cluster')

    with mock_teardown as teardown_cluster, \
         mock.patch('sky.backends.backend_utils'
                    '.refresh_cluster_status_handle',
                    return_value=(status_lib.ClusterStatus.UP,
                                 mock_handle)), \
         mock.patch('sky.global_user_state.remove_cluster'), \
         mock.patch('sky.global_user_state.get_cluster_yaml_dict',
                    return_value={'provider': {}}), \
         mock.patch.object(backend, 'post_teardown_cleanup'):

        start = time.time()
        backend.teardown_no_lock(mock_handle, terminate=True, purge=False)
        elapsed = time.time() - start

    assert elapsed < 15, (
        f'teardown_no_lock took {elapsed:.1f}s for a Slurm cluster. '
        f'Must complete quickly even when the poll worker is offline.')

    # Verify provisioner.teardown_cluster was called with the right
    # cluster name (it runs scancel --name={cluster_name_on_cloud}).
    # The actual scancel behavior is tested in the integration tests
    # (test_tunnel_integration.py::TestTerminateInstances).
    teardown_cluster.assert_called_once()
    call_args = teardown_cluster.call_args
    assert cluster_name in str(call_args), (
        f'teardown_cluster not called with cluster name {cluster_name}: '
        f'{call_args}')
