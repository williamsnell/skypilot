"""Regression tests for API server URL routing.

Internal callers (SkyServe controller, health checks) must connect
to the local API server, not the external authenticated endpoint.
External callers (poll workers on compute nodes) must get the
externally-reachable URL.

The bug: get_server_url() was reading SKYPILOT_API_SERVER_ENDPOINT,
returning the external URL for ALL callers. This caused the SkyServe
controller's sdk.launch() to hit the auth wall, breaking pool worker
provisioning.
"""
import subprocess
import sys


def test_health_check_connects_to_local_not_external():
    """check_server_healthy must connect to the local API server
    even when SKYPILOT_API_SERVER_ENDPOINT points to an external URL.

    Starts a mock server on localhost, sets the env var to an external
    URL, and calls check_server_healthy. If it connects to the local
    mock (200 OK), the test passes. If it tries the external URL,
    it will get a connection error or auth failure.
    """
    result = subprocess.run(
        [
            sys.executable, '-c', """
import json
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ['SKYPILOT_API_SERVER_ENDPOINT'] = 'https://sky.example.com'

# Start a mock API server on the default local port.
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/health':
            body = json.dumps({
                'status': 'healthy',
                'api_version': '1',
                'commit': 'test',
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args):
        pass  # Suppress logs.

server = HTTPServer(('127.0.0.1', 46580), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()

from sky.server import common
from sky import exceptions

try:
    status, info = common.check_server_healthy()
    print(f'OK: check_server_healthy returned {status}')
except exceptions.ApiServerAuthenticationError:
    print('FAIL: check_server_healthy hit the external auth wall')
    raise SystemExit(1)
except Exception as e:
    # Connection error to sky.example.com would show up here.
    print(f'FAIL: check_server_healthy tried external URL: {e}')
    raise SystemExit(1)
finally:
    server.shutdown()
"""
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f'check_server_healthy must connect to local server, not '
        f'external:\n{result.stdout.strip()}\n'
        f'{result.stderr.strip()[-500:]}')


def test_poll_worker_sbatch_config_uses_external_url():
    """The poll worker launch block in the sbatch script must contain
    the externally-reachable URL, not the local one.

    Calls _sbatch_keep_alive_block (which generates the actual sbatch
    script content) with an external URL and verifies it appears in
    the --api-server-url argument.
    """
    result = subprocess.run(
        [
            sys.executable, '-c', """
import os
os.environ['SKYPILOT_API_SERVER_ENDPOINT'] = 'https://sky.example.com'

from sky.server import common
from sky.provision.slurm.instance import _sbatch_keep_alive_block

# Get the URL the same way the provisioning code does.
url = common.get_server_global_url()

# Generate the sbatch keep-alive block with poll worker config.
block = _sbatch_keep_alive_block(
    container_image='test-image',
    proctrack_type='cgroup',
    container_runtime='podman-hpc',
    sky_cluster_home_dir='/root',
    skypilot_runtime_dir='/tmp/runtime',
    cluster_name_on_cloud='test-cluster',
    poll_token='fake-token',
    poll_pubkey_b64='ZmFrZQ==',
    api_server_url=url,
)

if 'https://sky.example.com' not in block:
    print(f'FAIL: sbatch script does not contain external URL')
    print(f'Block content: {block[:500]}')
    raise SystemExit(1)
if '--api-server-url' not in block:
    print('FAIL: sbatch script missing --api-server-url flag')
    raise SystemExit(1)
print('OK: poll worker config contains external URL')
"""
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'Poll worker sbatch config must contain external URL:\n'
        f'{result.stdout.strip()}\n{result.stderr.strip()[-500:]}')
