"""Fixtures for Slurm poll-based task queue integration tests.

Runs a minimal in-process FastAPI server that includes the production
poll task router (from slurm_task_queue.create_poll_router) in a
background thread. This ensures integration tests exercise the exact
same endpoint code as the production API server.

The Slurm cluster (slurmctld) provides SSH for sanity checking.
The poll worker subprocess connects to the in-process mini server.
"""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Generator

import pytest
import uvicorn

from sky.server.slurm_task_queue import create_poll_router
from sky.server.slurm_task_queue import get_task_queue
from sky.server.slurm_task_queue import SlurmTaskQueue

TEST_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Mini FastAPI server using the production poll task router
# ---------------------------------------------------------------------------

# Captured from the uvicorn thread so tests can schedule async calls.
_server_loop: asyncio.AbstractEventLoop = None  # type: ignore
_server_loop_ready = threading.Event()


def _create_mini_app():
    """Create a FastAPI app with the production poll task router."""
    import fastapi  # pylint: disable=import-outside-toplevel

    app = fastapi.FastAPI()
    app.include_router(create_poll_router())

    @app.on_event('startup')
    async def _capture_loop():
        global _server_loop
        _server_loop = asyncio.get_event_loop()
        _server_loop_ready.set()

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _is_in_container() -> bool:
    return os.path.exists('/.dockerenv') or os.environ.get(
        'SLURM_TUNNEL_TEST_CONTAINER') == '1'


@pytest.fixture(scope='session')
def mini_server():
    """Start the mini FastAPI server in a background thread."""
    app = _create_mini_app()
    port = 18932

    config = uvicorn.Config(app, host='0.0.0.0', port=port, log_level='info')
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    assert _server_loop_ready.wait(timeout=15), 'Server did not start'

    # Wait for server to accept connections
    import socket  # pylint: disable=import-outside-toplevel
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            s = socket.create_connection(('127.0.0.1', port), timeout=1)
            s.close()
            break
        except OSError:
            time.sleep(0.2)

    yield port

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope='session')
def api_server_url(mini_server) -> str:
    return f'http://127.0.0.1:{mini_server}'


@pytest.fixture(scope='session')
def task_queue() -> SlurmTaskQueue:
    """The module-level SlurmTaskQueue singleton shared with the server."""
    return get_task_queue()


@pytest.fixture(scope='session')
def run_async(mini_server):
    """Returns a helper to schedule coroutines on the server's event loop."""

    def _run(coro, timeout=15):
        assert _server_loop is not None, 'Mini server not started'
        import asyncio as _asyncio
        future = _asyncio.run_coroutine_threadsafe(coro, _server_loop)
        return future.result(timeout=timeout)

    return _run


@pytest.fixture(scope='session')
def ssh_key() -> Path:
    """Shared test SSH key."""
    key = TEST_DIR / '.test_keys' / 'id_ed25519'
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ['ssh-keygen', '-t', 'ed25519', '-f',
             str(key), '-N', '', '-q'],
            check=True)
    return key


@pytest.fixture(scope='session')
def slurm_cluster(ssh_key) -> dict:
    """Connection info for slurmctld's SSH."""
    if _is_in_container():
        return {
            'host': 'slurmctld',
            'port': 22,
            'user': 'root',
            'ssh_key': str(ssh_key),
        }
    return {
        'host': 'localhost',
        'port': 2222,
        'user': 'root',
        'ssh_key': str(ssh_key),
    }


@pytest.fixture(scope='session')
def cluster_credentials(task_queue) -> Generator[dict, None, None]:
    """Generate and register a poll token + signing keypair."""
    cluster_name = 'test-poll-cluster'
    token = task_queue.generate_token(cluster_name)
    task_queue.store_token(cluster_name, token)
    pubkey_bytes = task_queue.generate_keypair(cluster_name)

    # Write public key to a temp file for the poll worker
    pubkey_file = TEST_DIR / '.test_keys' / 'server_pubkey'
    pubkey_file.parent.mkdir(parents=True, exist_ok=True)
    pubkey_file.write_bytes(pubkey_bytes)

    yield {
        'cluster_name': cluster_name,
        'token': token,
        'pubkey_file': str(pubkey_file),
    }
    task_queue.cleanup_cluster(cluster_name)
    pubkey_file.unlink(missing_ok=True)


@pytest.fixture
def poll_worker_proc(cluster_credentials, api_server_url):
    """Start a poll worker subprocess connecting to the mini server."""
    env = os.environ.copy()
    env['SKYPILOT_POLL_TOKEN'] = cluster_credentials['token']
    env['SKYPILOT_POLL_ALLOW_HTTP'] = '1'
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'sky.provision.slurm.poll_worker',
            '--api-server-url',
            api_server_url,
            '--cluster-name',
            cluster_credentials['cluster_name'],
            '--server-pubkey-file',
            cluster_credentials['pubkey_file'],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
