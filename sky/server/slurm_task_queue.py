"""Task queue for polling-based Slurm worker communication.

The API server enqueues tasks (setup scripts, job codegen) and the worker
polls for them via REST endpoints. This replaces the reverse WebSocket
tunnel approach with a simpler, more secure model: the worker only makes
outbound HTTP calls, and the server signs every task payload so the worker
can verify authenticity.

Authentication is mutual:
    Worker → Server: random token (persisted in kv_cache)
    Server → Worker: Ed25519 signature on task payloads

All task state is persisted in kv_cache (backed by SQLite/PostgreSQL) so
that it is visible across the FastAPI server process and the backend
executor processes.

Usage (server-side):
    queue = get_task_queue()
    pubkey = queue.generate_keypair('my-cluster')
    token = queue.generate_token('my-cluster')
    queue.store_token('my-cluster', token)
    # ... pass pubkey + token to sbatch script ...

    task_id = queue.enqueue_task('my-cluster', TaskType.SETUP, script_content)
    exit_code, stdout, stderr = queue.wait_for_completion(
        'my-cluster', task_id, timeout=600)
"""
import enum
import hashlib
import json
import logging
import secrets
import threading
import time
from typing import Dict, Optional, Tuple
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import NoEncryption
from cryptography.hazmat.primitives.serialization import PrivateFormat
from cryptography.hazmat.primitives.serialization import PublicFormat

from sky.utils.db import kv_cache

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = 'slurm:poll_token:'
_HEARTBEAT_CACHE_PREFIX = 'poll_heartbeat:'
_HEARTBEAT_CACHE_TTL = 120.0  # seconds before kv_cache auto-expires entry

# Task kv_cache key prefixes.
_TASK_KEY_PREFIX = 'slurm:task:'
_TASK_RESULT_KEY_PREFIX = 'slurm:task_result:'
_SIGNING_KEY_PREFIX = 'slurm:signing_key:'

# Task entries expire after 4 hours (covers long-running setups).
_TASK_TTL = 4 * 3600
# Result entries expire after 1 hour.
_RESULT_TTL = 3600

# Heartbeat older than this is considered stale.
HEARTBEAT_TIMEOUT_SECONDS = 60.0

# Polling interval when waiting for task completion.
_COMPLETION_POLL_INTERVAL = 1.0


class TaskType(enum.Enum):
    SETUP = 'setup'
    RUN = 'run'


class TaskStatus(enum.Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'


class SlurmTaskQueue:
    """DB-backed task queue with mutual authentication.

    All task state is persisted in kv_cache so it is visible across
    the FastAPI server process and backend executor processes.
    """

    def __init__(self):
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Token management (worker → server auth, persisted in kv_cache)
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_token(cluster_name: str) -> str:  # pylint: disable=unused-argument
        """Generate a cryptographically secure token for a cluster."""
        return secrets.token_hex(32)

    @staticmethod
    def _hash_token(token: str) -> str:
        """SHA-256 hash a token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def store_token(cluster_name: str, token: str) -> None:
        """Persist a hashed token in the server's KV cache."""
        cache_key = f'{_CACHE_KEY_PREFIX}{cluster_name}'
        far_future = time.time() + 365 * 24 * 3600
        hashed = SlurmTaskQueue._hash_token(token)
        kv_cache.add_or_update_cache_entry(cache_key, hashed, far_future)

    @staticmethod
    def validate_token(cluster_name: str, token: str) -> bool:
        """Validate a token against the stored hash."""
        cache_key = f'{_CACHE_KEY_PREFIX}{cluster_name}'
        stored_hash = kv_cache.get_cache_entry(cache_key)
        if stored_hash is None:
            return False
        return secrets.compare_digest(stored_hash,
                                      SlurmTaskQueue._hash_token(token))

    @staticmethod
    def remove_tokens(cluster_name: str) -> None:
        """Invalidate tokens on cluster teardown."""
        cache_key = f'{_CACHE_KEY_PREFIX}{cluster_name}'
        try:
            kv_cache.add_or_update_cache_entry(cache_key, '', 0)
        except Exception:  # pylint: disable=broad-except
            logger.debug('Failed to remove poll token for %s', cluster_name)

    # ------------------------------------------------------------------ #
    # Signing key management (server → worker auth, persisted in kv_cache)
    # ------------------------------------------------------------------ #

    def generate_keypair(self, cluster_name: str) -> bytes:
        """Generate an Ed25519 keypair. Returns raw public key bytes.

        The private key is persisted in kv_cache so the FastAPI server
        process (which signs task payloads) can access it even when the
        keypair was generated in a different executor process.
        """
        private_key = Ed25519PrivateKey.generate()
        # Serialize private key for DB storage.
        raw_private = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw,
                                                NoEncryption())
        cache_key = f'{_SIGNING_KEY_PREFIX}{cluster_name}'
        kv_cache.add_or_update_cache_entry(cache_key, raw_private.hex(),
                                           time.time() + _TASK_TTL)
        public_key = private_key.public_key()
        return public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def _get_signing_key(self,
                         cluster_name: str) -> Optional[Ed25519PrivateKey]:
        """Load the signing key from kv_cache."""
        cache_key = f'{_SIGNING_KEY_PREFIX}{cluster_name}'
        raw_hex = kv_cache.get_cache_entry(cache_key)
        if raw_hex is None:
            return None
        return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw_hex))

    def sign_task(self, cluster_name: str, task_id: str, task_type: str,
                  script_content: str, nonce: str) -> str:
        """Sign a task's canonical payload. Returns hex-encoded signature."""
        key = self._get_signing_key(cluster_name)
        if key is None:
            raise ValueError(f'No signing key for cluster {cluster_name}')
        payload = _canonical_payload(task_id, task_type, script_content, nonce)
        signature = key.sign(payload)
        return signature.hex()

    def get_public_key_bytes(self, cluster_name: str) -> Optional[bytes]:
        """Get the raw public key bytes for a cluster."""
        key = self._get_signing_key(cluster_name)
        if key is None:
            return None
        return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    # ------------------------------------------------------------------ #
    # Task lifecycle (DB-backed)
    # ------------------------------------------------------------------ #

    def enqueue_task(
        self,
        cluster_name: str,
        task_type: TaskType,
        script_content: str,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> str:
        """Enqueue a task for the worker. Returns the task_id.

        Use wait_for_completion() to block until the worker reports back.
        """
        task_id = uuid.uuid4().hex[:12]
        task_data = {
            'task_id': task_id,
            'task_type': task_type.value,
            'script_content': script_content,
            'env_vars': env_vars or {},
            'status': TaskStatus.PENDING.value,
            'created_at': time.time(),
        }
        cache_key = f'{_TASK_KEY_PREFIX}{cluster_name}:{task_id}'
        kv_cache.add_or_update_cache_entry(cache_key, json.dumps(task_data),
                                           time.time() + _TASK_TTL)
        self._add_to_index(cluster_name, task_id)
        logger.info('Enqueued %s task %s for %s', task_type.value, task_id,
                    cluster_name)
        return task_id

    def dequeue_task(self, cluster_name: str,
                     nonce: str) -> Optional[Dict[str, object]]:
        """Get the next pending task as a serializable dict with signature.

        Returns None if no pending tasks. Marks the task as RUNNING.
        """
        # Look up pending tasks for this cluster by scanning known task keys.
        # Since tasks are processed one at a time (executor blocks on each),
        # there's typically 0 or 1 pending task.
        task_data = self._find_pending_task(cluster_name)
        if task_data is None:
            return None

        # Mark as RUNNING.
        task_data['status'] = TaskStatus.RUNNING.value
        cache_key = (f'{_TASK_KEY_PREFIX}{cluster_name}:'
                     f'{task_data["task_id"]}')
        kv_cache.add_or_update_cache_entry(cache_key, json.dumps(task_data),
                                           time.time() + _TASK_TTL)

        signature = self.sign_task(cluster_name, task_data['task_id'],
                                   task_data['task_type'],
                                   task_data['script_content'], nonce)
        return {
            'task_id': task_data['task_id'],
            'task_type': task_data['task_type'],
            'script_content': task_data['script_content'],
            'env_vars': task_data.get('env_vars', {}),
            'nonce': nonce,
            'signature': signature,
        }

    def _find_pending_task(self, cluster_name: str) -> Optional[dict]:
        """Find the oldest pending task for a cluster.

        Uses the task index key to look up task IDs without scanning.
        """
        index_key = f'{_TASK_KEY_PREFIX}{cluster_name}:__index__'
        index_raw = kv_cache.get_cache_entry(index_key)
        if not index_raw:
            return None
        task_ids = json.loads(index_raw)
        for task_id in task_ids:
            cache_key = f'{_TASK_KEY_PREFIX}{cluster_name}:{task_id}'
            raw = kv_cache.get_cache_entry(cache_key)
            if raw is None:
                continue
            task_data = json.loads(raw)
            if task_data.get('status') == TaskStatus.PENDING.value:
                return task_data
        return None

    def _add_to_index(self, cluster_name: str, task_id: str) -> None:
        """Add a task_id to the cluster's task index."""
        index_key = f'{_TASK_KEY_PREFIX}{cluster_name}:__index__'
        index_raw = kv_cache.get_cache_entry(index_key)
        task_ids = json.loads(index_raw) if index_raw else []
        task_ids.append(task_id)
        kv_cache.add_or_update_cache_entry(index_key, json.dumps(task_ids),
                                           time.time() + _TASK_TTL)

    def complete_task(
        self,
        cluster_name: str,
        task_id: str,
        exit_code: int,
        stdout: str = '',
        stderr: str = '',
    ) -> bool:
        """Mark a task as completed and write the result to kv_cache.

        Returns True if the task was found, False otherwise.
        """
        cache_key = f'{_TASK_KEY_PREFIX}{cluster_name}:{task_id}'
        raw = kv_cache.get_cache_entry(cache_key)
        if raw is None:
            logger.warning('complete_task: unknown task %s for %s', task_id,
                           cluster_name)
            return False

        # Update task status.
        task_data = json.loads(raw)
        task_data['status'] = (TaskStatus.COMPLETED.value
                               if exit_code == 0 else TaskStatus.FAILED.value)
        kv_cache.add_or_update_cache_entry(cache_key, json.dumps(task_data),
                                           time.time() + _RESULT_TTL)

        # Write result to a separate key that wait_for_completion polls.
        result_key = f'{_TASK_RESULT_KEY_PREFIX}{cluster_name}:{task_id}'
        result_data = {
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr,
        }
        kv_cache.add_or_update_cache_entry(result_key, json.dumps(result_data),
                                           time.time() + _RESULT_TTL)

        logger.info('Task %s for %s completed (exit_code=%d)', task_id,
                    cluster_name, exit_code)
        return True

    def wait_for_completion(
        self,
        cluster_name: str,
        task_id: str,
        timeout: float = 3600,
    ) -> Tuple[int, str, str]:
        """Poll kv_cache until the task result appears.

        Returns (exit_code, stdout, stderr).
        Raises TimeoutError if the deadline is exceeded.
        Raises RuntimeError if the poll worker goes offline.
        """
        result_key = f'{_TASK_RESULT_KEY_PREFIX}{cluster_name}:{task_id}'
        deadline = time.time() + timeout
        checks_since_online = 0
        while time.time() < deadline:
            raw = kv_cache.get_cache_entry(result_key)
            if raw is not None:
                result = json.loads(raw)
                return (result['exit_code'], result['stdout'], result['stderr'])
            # Periodically check worker liveness so we fail fast when
            # the poll worker is dead rather than blocking for the full
            # timeout.  Check every few iterations to avoid DB spam.
            checks_since_online += 1
            if checks_since_online >= 5:
                if not self.is_worker_online(cluster_name):
                    raise RuntimeError(
                        f'Poll worker for {cluster_name} is offline '
                        f'(no heartbeat). Task {task_id} will never '
                        f'complete.')
                checks_since_online = 0
            time.sleep(_COMPLETION_POLL_INTERVAL)
        raise TimeoutError(
            f'Task {task_id} for {cluster_name} timed out after {timeout}s')

    # ------------------------------------------------------------------ #
    # Heartbeat
    # ------------------------------------------------------------------ #

    def record_heartbeat(self, cluster_name: str) -> None:
        """Record a heartbeat from the poll worker."""
        now = time.time()
        kv_cache.add_or_update_cache_entry(
            f'{_HEARTBEAT_CACHE_PREFIX}{cluster_name}',
            str(now),
            expires_at=now + _HEARTBEAT_CACHE_TTL)

    def is_worker_online(
        self,
        cluster_name: str,
        timeout: float = HEARTBEAT_TIMEOUT_SECONDS,
    ) -> bool:
        """Check if the poll worker has heartbeated recently."""
        entry = kv_cache.get_cache_entry(
            f'{_HEARTBEAT_CACHE_PREFIX}{cluster_name}')
        if entry is None:
            return False
        return (time.time() - float(entry)) < timeout

    def get_last_heartbeat(self, cluster_name: str) -> Optional[float]:
        """Return the timestamp of the last heartbeat, or None."""
        entry = kv_cache.get_cache_entry(
            f'{_HEARTBEAT_CACHE_PREFIX}{cluster_name}')
        if entry is None:
            return None
        return float(entry)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def cleanup_cluster(self, cluster_name: str) -> None:
        """Remove all state for a cluster (on teardown)."""
        # Cancel any pending task by writing a cancellation result.
        pending = self._find_pending_task(cluster_name)
        if pending:
            self.complete_task(cluster_name,
                               pending['task_id'],
                               exit_code=-1,
                               stderr='Task cancelled: cluster teardown')

        # Invalidate tokens, signing keys, heartbeat.
        self.remove_tokens(cluster_name)
        _remove_cache_entry(f'{_SIGNING_KEY_PREFIX}{cluster_name}')
        _remove_cache_entry(f'{_HEARTBEAT_CACHE_PREFIX}{cluster_name}')
        _remove_cache_entry(f'{_TASK_KEY_PREFIX}{cluster_name}:__index__')
        logger.info('Cleaned up state for cluster %s', cluster_name)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _remove_cache_entry(key: str) -> None:
    """Effectively remove a kv_cache entry by setting expires_at to 0."""
    try:
        kv_cache.add_or_update_cache_entry(key, '', expires_at=0)
    except Exception:  # pylint: disable=broad-except
        pass


def _canonical_payload(task_id: str, task_type: str, script_content: str,
                       nonce: str) -> bytes:
    """Deterministic byte string for signing/verification."""
    return f'{task_id}:{task_type}:{script_content}:{nonce}'.encode('utf-8')


# ---------------------------------------------------------------------- #
# FastAPI router (shared by production server.py and integration tests)
# ---------------------------------------------------------------------- #


def create_poll_router():
    """Create a FastAPI APIRouter with the Slurm poll task endpoints.

    Used by sky.server.server (production) and integration tests so both
    exercise the exact same endpoint code.
    """
    import fastapi  # pylint: disable=import-outside-toplevel

    router = fastapi.APIRouter()

    def _check_token(cluster_name: str, token: str):
        if not SlurmTaskQueue.validate_token(cluster_name, token):
            raise fastapi.HTTPException(status_code=401, detail='Invalid token')

    @router.get('/slurm/tasks/{cluster_name}')
    async def slurm_poll_tasks(
            cluster_name: str,
            token: str = fastapi.Header(..., alias='X-Slurm-Token'),
            nonce: str = fastapi.Header(..., alias='X-Slurm-Nonce'),
    ) -> dict:
        """Poll worker fetches the next pending task."""
        _check_token(cluster_name, token)
        task = get_task_queue().dequeue_task(cluster_name, nonce=nonce)
        return {'task': task}

    @router.post('/slurm/tasks/{cluster_name}/{task_id}/result')
    async def slurm_task_result(
            cluster_name: str,
            task_id: str,
            token: str = fastapi.Header(..., alias='X-Slurm-Token'),
            body: dict = fastapi.Body(...),
    ) -> dict:
        """Poll worker reports task completion."""
        _check_token(cluster_name, token)
        found = get_task_queue().complete_task(
            cluster_name,
            task_id,
            exit_code=body.get('exit_code', -1),
            stdout=body.get('stdout', ''),
            stderr=body.get('stderr', ''),
        )
        if not found:
            raise fastapi.HTTPException(status_code=404, detail='Unknown task')
        return {'status': 'ok'}

    @router.post('/slurm/tasks/{cluster_name}/heartbeat')
    async def slurm_heartbeat(
            cluster_name: str,
            token: str = fastapi.Header(..., alias='X-Slurm-Token'),
    ) -> dict:
        """Poll worker heartbeat — indicates the container is alive."""
        _check_token(cluster_name, token)
        get_task_queue().record_heartbeat(cluster_name)
        return {'status': 'ok'}

    return router


# ---------------------------------------------------------------------- #
# Module-level singleton
# ---------------------------------------------------------------------- #

_queue: Optional[SlurmTaskQueue] = None
_queue_lock = threading.Lock()


def get_task_queue() -> SlurmTaskQueue:
    """Get (or create) the global SlurmTaskQueue singleton."""
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = SlurmTaskQueue()
    return _queue
