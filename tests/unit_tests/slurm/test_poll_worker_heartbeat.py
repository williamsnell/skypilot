"""Integration test: poll worker → FastAPI server → kv_cache DB.

Starts a real uvicorn server and poll worker in separate processes,
verifying that heartbeats are persisted to the kv_cache DB and
visible via the production _wait_for_poll_worker code.
"""
import multiprocessing
import os
import secrets
import socket
import tempfile
import time

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _run_server(port: int, cluster_name: str, token: str, pubkey_bytes: bytes,
                ready_event, stop_event):
    """Start a minimal FastAPI app with the poll router."""
    import fastapi
    import uvicorn

    from sky.server.slurm_task_queue import create_poll_router
    from sky.server.slurm_task_queue import get_task_queue

    # Set up queue state for this cluster.
    queue = get_task_queue()
    queue.store_token(cluster_name, token)
    queue.generate_keypair(cluster_name)

    app = fastapi.FastAPI()
    app.include_router(create_poll_router())

    # Health endpoint for poll worker connectivity test.
    @app.get('/api/health')
    async def health():
        return {'status': 'ok'}

    config = uvicorn.Config(app,
                            host='127.0.0.1',
                            port=port,
                            log_level='warning')
    server = uvicorn.Server(config)

    # Signal that the server is starting.
    ready_event.set()
    server.run()


def _run_poll_worker(port: int, cluster_name: str, token: str,
                     pubkey_file: str):
    """Run the poll worker, allowing HTTP for localhost testing."""
    os.environ['SKYPILOT_POLL_ALLOW_HTTP'] = '1'
    os.environ['SKYPILOT_POLL_TOKEN'] = token

    from sky.provision.slurm.poll_worker import _load_public_key
    from sky.provision.slurm.poll_worker import run

    pubkey = _load_public_key(pubkey_file)
    run(f'http://127.0.0.1:{port}', cluster_name, token, pubkey)


def _delayed_heartbeat_writer(cache_key: str, delay: float = 2.0):
    """Write a heartbeat after a delay (for mid-wait tests)."""
    time.sleep(delay)
    from sky.utils.db import kv_cache
    kv_cache.add_or_update_cache_entry(cache_key,
                                       str(time.time()),
                                       expires_at=time.time() + 120)


class TestWaitForPollWorker:
    """Test the actual _wait_for_poll_worker production code."""

    def test_returns_when_heartbeat_present(self):
        """_wait_for_poll_worker returns immediately when heartbeat exists."""
        from sky.backends.cloud_vm_ray_backend import CloudVmRayBackend
        from sky.utils.db import kv_cache

        cluster_name = f'wait-test-{secrets.token_hex(4)}'
        cache_key = f'poll_heartbeat:{cluster_name}'

        # Write a heartbeat to the DB (simulating the server endpoint).
        kv_cache.add_or_update_cache_entry(cache_key,
                                           str(time.time()),
                                           expires_at=time.time() + 120)

        # Create a minimal mock handle.
        handle = type('Handle', (), {
            'cluster_name_on_cloud': cluster_name,
        })()

        backend = CloudVmRayBackend()
        # Should return quickly (not wait 120s).
        start = time.time()
        backend._wait_for_poll_worker(handle, timeout=5.0)
        elapsed = time.time() - start
        assert elapsed < 3, f'Took {elapsed:.1f}s, expected < 3s'

    def test_raises_on_timeout(self):
        """_wait_for_poll_worker raises RuntimeError after timeout."""
        from sky.backends.cloud_vm_ray_backend import CloudVmRayBackend

        cluster_name = f'timeout-test-{secrets.token_hex(4)}'

        handle = type(
            'Handle', (), {
                'cluster_name_on_cloud': cluster_name,
                'get_command_runners': lambda self: [],
            })()

        backend = CloudVmRayBackend()
        with pytest.raises(RuntimeError, match='did not come online'):
            backend._wait_for_poll_worker(handle, timeout=3.0)

    def test_returns_when_heartbeat_arrives_mid_wait(self):
        """_wait_for_poll_worker detects heartbeat arriving during poll."""
        from sky.backends.cloud_vm_ray_backend import CloudVmRayBackend

        cluster_name = f'delayed-test-{secrets.token_hex(4)}'
        cache_key = f'poll_heartbeat:{cluster_name}'

        handle = type('Handle', (), {
            'cluster_name_on_cloud': cluster_name,
        })()

        # Write heartbeat after a short delay in another process.
        writer = multiprocessing.Process(target=_delayed_heartbeat_writer,
                                         args=(cache_key, 2.0))
        writer.start()

        backend = CloudVmRayBackend()
        start = time.time()
        backend._wait_for_poll_worker(handle, timeout=15.0)
        elapsed = time.time() - start

        writer.join()
        # Should return after ~2-4s (delay + poll interval), not 15s.
        assert 1.5 < elapsed < 8, f'Took {elapsed:.1f}s, expected 2-6s'


class TestPollWorkerHeartbeatE2E:
    """End-to-end: poll worker → server → kv_cache → _wait_for_poll_worker.

    Starts a real FastAPI server and real poll worker in separate
    processes, then calls the actual _wait_for_poll_worker production
    code to verify the heartbeat is detected.
    """

    def test_poll_worker_heartbeat_detected_by_wait(self):
        from sky.backends.cloud_vm_ray_backend import CloudVmRayBackend

        cluster_name = f'e2e-test-{secrets.token_hex(4)}'
        port = _free_port()

        # Generate auth materials.
        from sky.server.slurm_task_queue import get_task_queue
        queue = get_task_queue()
        token = queue.generate_token(cluster_name)
        queue.store_token(cluster_name, token)
        pubkey_bytes = queue.generate_keypair(cluster_name)

        # Write pubkey to a temp file for the poll worker.
        pubkey_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pub')
        pubkey_file.write(pubkey_bytes)
        pubkey_file.close()

        ready_event = multiprocessing.Event()
        stop_event = multiprocessing.Event()

        # Start server (separate process — simulates uvicorn worker).
        server_proc = multiprocessing.Process(target=_run_server,
                                              args=(port, cluster_name, token,
                                                    pubkey_bytes, ready_event,
                                                    stop_event))
        server_proc.start()
        ready_event.wait(timeout=10)
        time.sleep(0.5)

        # Start poll worker (separate process — simulates Slurm container).
        worker_proc = multiprocessing.Process(target=_run_poll_worker,
                                              args=(port, cluster_name, token,
                                                    pubkey_file.name))
        worker_proc.start()

        try:
            # Call the actual production _wait_for_poll_worker.
            # This runs in the test process (simulates request executor).
            handle = type(
                'Handle', (), {
                    'cluster_name_on_cloud': cluster_name,
                    'get_command_runners': lambda self: [],
                })()

            backend = CloudVmRayBackend()
            start = time.time()
            backend._wait_for_poll_worker(handle, timeout=15.0)
            elapsed = time.time() - start

            # Poll worker sends heartbeat immediately on startup,
            # so this should complete well within the timeout.
            assert elapsed < 10, f'Took {elapsed:.1f}s, expected < 10s'
        finally:
            worker_proc.terminate()
            worker_proc.join(timeout=5)
            server_proc.terminate()
            server_proc.join(timeout=5)
            os.unlink(pubkey_file.name)
