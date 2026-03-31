"""Polling-based task worker for Slurm + podman-hpc clusters.

Runs inside the container on a Slurm compute node (launched from the
sbatch batch step cgroup alongside Skylet). Polls the SkyPilot API
server for tasks, executes them locally, and reports results.

Security model:
    Worker → Server: random token in X-Slurm-Token header
    Server → Worker: Ed25519 signature on every task payload;
        worker verifies before executing.

Usage:
    export SKYPILOT_POLL_TOKEN=<hex-token>
    python -m sky.provision.slurm.poll_worker \\
        --api-server-url https://api-server:46580 \\
        --cluster-name my-cluster \\
        --server-pubkey-file /root/.sky/.server_pubkey

HTTPS is required. Set SKYPILOT_POLL_ALLOW_HTTP=1 to override for testing.

The token is read from the SKYPILOT_POLL_TOKEN environment variable
(preferred, not visible in process listings) or --token (fallback).
"""
import argparse
import json
import logging
import os
import secrets
import subprocess
import tempfile
import time
from typing import Optional
import urllib.error
import urllib.request

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger('sky.poll_worker')

_POLL_INTERVAL = 2  # seconds between task polls
_HEARTBEAT_INTERVAL = 30  # seconds between heartbeats
_INITIAL_BACKOFF = 5  # seconds on HTTP error
_MAX_BACKOFF = 60  # max backoff


def _canonical_payload(task_id: str, task_type: str, script_content: str,
                       nonce: str) -> bytes:
    """Must match server-side _canonical_payload."""
    return f'{task_id}:{task_type}:{script_content}:{nonce}'.encode('utf-8')


def _load_public_key(path: str) -> Ed25519PublicKey:
    """Load a raw Ed25519 public key (32 bytes) from a file."""
    with open(path, 'rb') as f:
        raw_bytes = f.read()
    return Ed25519PublicKey.from_public_bytes(raw_bytes)


def _verify_signature(pubkey: Ed25519PublicKey, task: dict,
                      expected_nonce: str) -> bool:
    """Verify the server's Ed25519 signature on a task payload.

    The signature must cover the nonce we sent in the poll request.
    Also checks that the returned nonce matches what we sent.
    """
    sig_hex = task.get('signature', '')
    if not sig_hex:
        logger.warning('Task %s has no signature', task.get('task_id'))
        return False
    returned_nonce = task.get('nonce', '')
    if returned_nonce != expected_nonce:
        logger.warning('Task %s nonce mismatch: expected=%s got=%s',
                       task.get('task_id'), expected_nonce, returned_nonce)
        return False
    payload = _canonical_payload(task['task_id'], task['task_type'],
                                 task['script_content'], expected_nonce)
    try:
        pubkey.verify(bytes.fromhex(sig_hex), payload)
        return True
    except Exception:  # pylint: disable=broad-except
        logger.warning('Signature verification failed for task %s',
                       task.get('task_id'))
        return False


def _require_https(url: str) -> None:
    """Reject non-HTTPS URLs unless explicitly overridden for testing."""
    if url.startswith('https://'):
        return
    if os.environ.get('SKYPILOT_POLL_ALLOW_HTTP') == '1':
        logger.warning('Using HTTP (SKYPILOT_POLL_ALLOW_HTTP=1) — '
                       'token and payloads are NOT encrypted')
        return
    raise ValueError(
        f'Poll worker requires HTTPS (got {url.split("/")[0]}//...). '
        f'Set SKYPILOT_POLL_ALLOW_HTTP=1 to override for testing.')


def _http_request(url: str,
                  token: str,
                  method: str = 'GET',
                  data: Optional[dict] = None,
                  extra_headers: Optional[dict] = None) -> dict:
    """Make an HTTP request with the poll token."""
    _require_https(url)
    headers = {
        'X-Slurm-Token': token,
        # Pass ingress-level auth (e.g. oauth2-proxy) which skips
        # requests with Bearer sky_* tokens. The actual authentication
        # is done by the X-Slurm-Token header at the endpoint level.
        'Authorization': 'Bearer sky_slurm_poll',
        # Override default Python-urllib User-Agent which is blocked by
        # Cloudflare's bot detection (error 1010).
        'User-Agent': 'skypilot-poll-worker/1.0',
    }
    if extra_headers:
        headers.update(extra_headers)
    body = None
    if data is not None:
        body = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _poll_for_task(base_url: str, cluster_name: str, token: str) -> tuple:
    """GET /slurm/tasks/{cluster_name} — returns (task_dict, nonce).

    Generates a random nonce per request. The server includes it in the
    Ed25519 signature, binding the response to this specific poll.
    """
    nonce = secrets.token_hex(16)
    url = f'{base_url}/slurm/tasks/{cluster_name}'
    resp = _http_request(url,
                         token,
                         method='GET',
                         extra_headers={'X-Slurm-Nonce': nonce})
    return resp.get('task'), nonce


def _report_result(base_url: str, cluster_name: str, task_id: str, token: str,
                   exit_code: int, stdout: str, stderr: str) -> None:
    """POST /slurm/tasks/{cluster_name}/{task_id}/result"""
    url = f'{base_url}/slurm/tasks/{cluster_name}/{task_id}/result'
    _http_request(url,
                  token,
                  method='POST',
                  data={
                      'exit_code': exit_code,
                      'stdout': stdout,
                      'stderr': stderr,
                  })


def _send_heartbeat(base_url: str, cluster_name: str, token: str) -> None:
    """POST /slurm/tasks/{cluster_name}/heartbeat"""
    url = f'{base_url}/slurm/tasks/{cluster_name}/heartbeat'
    _http_request(url, token, method='POST', data={})


def _execute_task(task: dict) -> tuple:
    """Write script to temp file, execute it, and return results.

    Returns (exit_code, stdout, stderr).
    """
    env = os.environ.copy()
    env.update(task.get('env_vars', {}))

    with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.sh',
            delete=False,
            prefix=f'sky_task_{task["task_id"]}_') as f:
        f.write(task['script_content'])
        script_path = f.name

    try:
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            ['bash', '-i', script_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, '', 'Task timed out after 3600 seconds'
    except Exception as e:  # pylint: disable=broad-except
        return -1, '', str(e)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def run(api_server_url: str, cluster_name: str, token: str,
        server_pubkey: Ed25519PublicKey) -> None:
    """Main poll loop. Runs until the process is killed."""
    logger.info('Poll worker starting: cluster=%s server=%s', cluster_name,
                api_server_url)

    last_heartbeat = 0.0
    backoff = _INITIAL_BACKOFF

    while True:
        try:
            # Heartbeat
            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                _send_heartbeat(api_server_url, cluster_name, token)
                last_heartbeat = time.time()
                logger.debug('Heartbeat sent')

            # Poll for task
            task, nonce = _poll_for_task(api_server_url, cluster_name, token)
            if task is not None:
                task_id = task['task_id']
                task_type = task['task_type']
                logger.info('Received task %s (type=%s)', task_id, task_type)

                # Verify server signature (covers our nonce) before executing
                if not _verify_signature(server_pubkey, task, nonce):
                    logger.error('REJECTING task %s: invalid signature',
                                 task_id)
                    _report_result(api_server_url,
                                   cluster_name,
                                   task_id,
                                   token,
                                   exit_code=-1,
                                   stdout='',
                                   stderr='Signature verification failed')
                    continue

                # Execute
                exit_code, stdout, stderr = _execute_task(task)
                logger.info('Task %s completed (exit_code=%d)', task_id,
                            exit_code)

                # Report result
                _report_result(api_server_url, cluster_name, task_id, token,
                               exit_code, stdout, stderr)

                backoff = _INITIAL_BACKOFF
            else:
                time.sleep(_POLL_INTERVAL)
                backoff = _INITIAL_BACKOFF

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logger.warning('HTTP error (backoff=%ds): %s', backoff, e)
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)
        except Exception as e:  # pylint: disable=broad-except
            logger.error('Unexpected error: %s', e, exc_info=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)


def main():
    parser = argparse.ArgumentParser(description='SkyPilot Slurm poll worker')
    parser.add_argument('--api-server-url',
                        required=True,
                        help='URL of the SkyPilot API server')
    parser.add_argument('--cluster-name',
                        required=True,
                        help='Slurm cluster name')
    parser.add_argument('--token',
                        default=None,
                        help='Authentication token (hex). '
                        'Prefer SKYPILOT_POLL_TOKEN env var.')
    parser.add_argument('--server-pubkey-file',
                        required=True,
                        help='Path to server Ed25519 public key (32 bytes)')
    args = parser.parse_args()

    # Prefer env var (not visible in ps) over CLI arg.
    token = os.environ.get('SKYPILOT_POLL_TOKEN') or args.token
    if not token:
        parser.error('Token required via SKYPILOT_POLL_TOKEN env var '
                     'or --token flag')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    pubkey = _load_public_key(args.server_pubkey_file)
    run(args.api_server_url, args.cluster_name, token, pubkey)


if __name__ == '__main__':
    main()
