"""Integration tests for the full remote API server architecture.

Runs `sky launch` through a real SkyPilot API server against a Slurm
cluster with mock podman-hpc. Validates the complete security model:

1. Client sends ephemeral SSH credentials → API server provisions
2. Poll-based operations (setup, exec) work without SSH
3. sky down works with fresh credentials (scancel)
4. Network isolation: testrunner cannot reach slurmctld or workers

Network topology:
    slurm_internal:  slurmctld, cpu-workers  (no API server)
    login_api:       slurmctld ↔ api-server  (SSH, ephemeral)
    poll_net:        api-server ↔ cpu-workers, testrunner-remote  (HTTP only)

Requires the 'remote' profile:

    podman compose -f docker-compose.yml --profile remote up -d --build
    podman compose -f docker-compose.yml --profile remote run --rm \
        testrunner-remote \
        python -m pytest tests/integration_tests/slurm_tunnel/test_remote_api.py -v
"""
import json
import os
import subprocess
import urllib.error
import urllib.request

import pytest

pytestmark = [
    pytest.mark.integration,
]

API_SERVER_URL = os.environ.get('SKYPILOT_REMOTE_API_URL',
                                'http://api-server:46580')


def _sky_env():
    """Environment variables for sky CLI pointing at the remote server."""
    env = os.environ.copy()
    env['SKYPILOT_API_SERVER_ENDPOINT'] = API_SERVER_URL
    env['SKYPILOT_DEV'] = '1'
    return env


def _sky(args, timeout=300):
    """Run a sky CLI command and return the result."""
    return subprocess.run(['python', '-m', 'sky.cli'] + args,
                          capture_output=True,
                          text=True,
                          timeout=timeout,
                          env=_sky_env())


def _api_get(path, timeout=10):
    """GET request to the remote API server."""
    url = f'{API_SERVER_URL}{path}'
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _write_task(path, run_cmd, image='mock-image'):
    """Write a minimal Slurm task YAML."""
    with open(path, 'w') as f:
        f.write('resources:\n')
        f.write('  cloud: slurm\n')
        f.write('  cpus: 1\n')
        f.write('  memory: 1\n')
        if image:
            f.write(f'  image_id: {image}\n')
        f.write(f'run: {run_cmd}\n')


class TestNetworkIsolation:
    """Verify the network topology enforces the expected constraints."""

    def test_api_server_reachable(self):
        """testrunner-remote can reach the API server on poll_net."""
        resp = _api_get('/api/health')
        assert 'status' in resp

    def test_cannot_reach_slurmctld(self):
        """testrunner-remote CANNOT reach slurmctld (different network)."""
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', '-o',
            'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=3', '-i',
            '/tmp/test_keys/id_ed25519', 'root@slurmctld', 'echo',
            'should-not-work'
        ],
                                capture_output=True,
                                text=True,
                                timeout=10)
        assert result.returncode != 0, (
            'testrunner should NOT reach slurmctld — network isolation broken')

    def test_cannot_reach_workers(self):
        """testrunner-remote CANNOT reach cpu-workers directly."""
        result = subprocess.run([
            'python', '-c', 'import socket; socket.create_connection('
            '("cpu-worker", 22), timeout=3)'
        ],
                                capture_output=True,
                                text=True,
                                timeout=10)
        assert result.returncode != 0


class TestClusterLifecycle:
    """Test the full cluster lifecycle: launch → exec → down.

    Each step uses ephemeral credentials sent from the client.
    The API server has NO local SSH key — it must use the credentials
    from the request.
    """

    def test_launch(self):
        """sky launch provisions via ephemeral SSH + executes via poll."""
        _write_task('/tmp/test_launch.yaml', 'echo "launch-ok"')
        result = _sky([
            'launch', '-y', '--cluster', 'test-lifecycle',
            '/tmp/test_launch.yaml'
        ])
        assert result.returncode == 0, (f'sky launch failed:\n'
                                        f'stdout: {result.stdout[-2000:]}\n'
                                        f'stderr: {result.stderr[-2000:]}')

    def test_exec_after_launch(self):
        """sky exec runs a command via poll queue (no SSH needed)."""
        _write_task('/tmp/test_exec.yaml', 'echo "exec-ok"')
        result = _sky(['exec', 'test-lifecycle', '/tmp/test_exec.yaml'],
                      timeout=120)
        assert result.returncode == 0, (f'sky exec failed:\n'
                                        f'stdout: {result.stdout[-2000:]}\n'
                                        f'stderr: {result.stderr[-2000:]}')

    def test_down(self):
        """sky down tears down with ephemeral SSH creds (scancel)."""
        result = _sky(['down', '-y', 'test-lifecycle'], timeout=60)
        assert result.returncode == 0, (f'sky down failed:\n'
                                        f'stdout: {result.stdout[-2000:]}\n'
                                        f'stderr: {result.stderr[-2000:]}')

    def test_down_is_clean(self):
        """After sky down, the cluster is gone."""
        result = _sky(['status', 'test-lifecycle'], timeout=30)
        # Cluster should not appear or should show as terminated
        assert 'test-lifecycle' not in result.stdout or \
            'TERMINATED' in result.stdout or \
            result.returncode != 0
