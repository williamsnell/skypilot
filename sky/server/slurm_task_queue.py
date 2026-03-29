"""Task queue for polling-based Slurm worker communication.

The API server enqueues tasks (setup scripts, job codegen) and the worker
polls for them via REST endpoints. This replaces the reverse WebSocket
tunnel approach with a simpler, more secure model: the worker only makes
outbound HTTP calls, and the server signs every task payload so the worker
can verify authenticity.

Authentication is mutual:
    Worker → Server: random token (persisted in kv_cache)
    Server → Worker: Ed25519 signature on task payloads

Usage (server-side):
    queue = get_task_queue()
    pubkey = queue.generate_keypair('my-cluster')
    token = queue.generate_token('my-cluster')
    queue.store_token('my-cluster', token)
    # ... pass pubkey + token to sbatch script ...

    future = queue.enqueue_task('my-cluster', TaskType.SETUP, script_content)
    result = future.result(timeout=600)  # blocks until worker completes
"""
import concurrent.futures
from dataclasses import dataclass
from dataclasses import field
import enum
import hashlib
import logging
import os
import secrets
import threading
import time
from typing import Dict, List, Optional, Tuple
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat

from sky.utils.db import kv_cache

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = 'slurm:poll_token:'

# Heartbeat older than this is considered stale.
HEARTBEAT_TIMEOUT_SECONDS = 60.0


class TaskType(enum.Enum):
    SETUP = 'setup'
    RUN = 'run'


class TaskStatus(enum.Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass
class SlurmTask:
    """A task queued for execution on a Slurm worker."""
    task_id: str
    task_type: TaskType
    script_content: str
    env_vars: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING

    # Populated on completion.
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    # Signaled when the worker reports completion.
    completion_future: concurrent.futures.Future = field(
        default_factory=concurrent.futures.Future)


@dataclass
class WorkerState:
    """Tracks a connected poll worker."""
    cluster_name: str
    last_heartbeat: float = field(default_factory=time.time)
    connected_at: float = field(default_factory=time.time)


class SlurmTaskQueue:
    """In-memory task queue with mutual authentication.

    Thread safety: all mutation is protected by a threading.Lock since
    enqueue_task is called from backend threads while dequeue/complete
    are called from the FastAPI event loop (via async endpoints that
    delegate to sync helpers).
    """

    def __init__(self):
        # cluster_name -> ordered list of SlurmTask
        self._tasks: Dict[str, List[SlurmTask]] = {}
        # cluster_name -> WorkerState
        self._workers: Dict[str, WorkerState] = {}
        # cluster_name -> Ed25519PrivateKey
        self._signing_keys: Dict[str, Ed25519PrivateKey] = {}
        # cluster_name -> list of temp file paths for ephemeral SSH keys
        self._ephemeral_key_files: Dict[str, List[str]] = {}
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
        """Invalidate tokens on cluster teardown.

        kv_cache has no delete API, so we set the value to empty with
        expiry 0 to effectively invalidate it. validate_token() will
        fail because compare_digest('', any_token) is always False.
        """
        cache_key = f'{_CACHE_KEY_PREFIX}{cluster_name}'
        try:
            kv_cache.add_or_update_cache_entry(cache_key, '', 0)
        except Exception:  # pylint: disable=broad-except
            logger.debug('Failed to remove poll token for %s', cluster_name)

    # ------------------------------------------------------------------ #
    # Signing key management (server → worker auth, in-memory)
    # ------------------------------------------------------------------ #

    def generate_keypair(self, cluster_name: str) -> bytes:
        """Generate an Ed25519 keypair. Returns raw public key bytes."""
        private_key = Ed25519PrivateKey.generate()
        with self._lock:
            self._signing_keys[cluster_name] = private_key
        public_key = private_key.public_key()
        return public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def sign_task(self, cluster_name: str, task: SlurmTask, nonce: str) -> str:
        """Sign a task's canonical payload. Returns hex-encoded signature."""
        with self._lock:
            key = self._signing_keys.get(cluster_name)
        if key is None:
            raise ValueError(f'No signing key for cluster {cluster_name}')
        payload = _canonical_payload(task.task_id, task.task_type.value,
                                     task.script_content, nonce)
        signature = key.sign(payload)
        return signature.hex()

    def get_public_key_bytes(self, cluster_name: str) -> Optional[bytes]:
        """Get the raw public key bytes for a cluster."""
        with self._lock:
            key = self._signing_keys.get(cluster_name)
        if key is None:
            return None
        return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    # ------------------------------------------------------------------ #
    # Task lifecycle
    # ------------------------------------------------------------------ #

    def enqueue_task(
        self,
        cluster_name: str,
        task_type: TaskType,
        script_content: str,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> concurrent.futures.Future:
        """Enqueue a task for the worker. Returns a Future resolved on
        completion with a (exit_code, stdout, stderr) tuple."""
        task = SlurmTask(
            task_id=uuid.uuid4().hex[:12],
            task_type=task_type,
            script_content=script_content,
            env_vars=env_vars or {},
        )
        with self._lock:
            self._tasks.setdefault(cluster_name, []).append(task)
        logger.info('Enqueued %s task %s for %s', task_type.value, task.task_id,
                    cluster_name)
        return task.completion_future

    def dequeue_task(self, cluster_name: str,
                     nonce: str) -> Optional[Dict[str, object]]:
        """Get the next pending task as a serializable dict with signature.

        Returns None if no pending tasks. Marks the task as RUNNING.
        The nonce (from the worker's poll request) is included in the
        signature so the response cannot be replayed.
        """
        with self._lock:
            tasks = self._tasks.get(cluster_name, [])
            task = None
            for t in tasks:
                if t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.RUNNING
                    task = t
                    break
        if task is None:
            return None
        signature = self.sign_task(cluster_name, task, nonce)
        return {
            'task_id': task.task_id,
            'task_type': task.task_type.value,
            'script_content': task.script_content,
            'env_vars': task.env_vars,
            'nonce': nonce,
            'signature': signature,
        }

    def complete_task(
        self,
        cluster_name: str,
        task_id: str,
        exit_code: int,
        stdout: str = '',
        stderr: str = '',
    ) -> bool:
        """Mark a task as completed and resolve its Future.

        Returns True if the task was found, False otherwise.
        """
        with self._lock:
            tasks = self._tasks.get(cluster_name, [])
            task = None
            for t in tasks:
                if t.task_id == task_id:
                    task = t
                    break
            if task is None:
                logger.warning('complete_task: unknown task %s for %s', task_id,
                               cluster_name)
                return False
            task.exit_code = exit_code
            task.stdout = stdout
            task.stderr = stderr
            task.status = (TaskStatus.COMPLETED
                           if exit_code == 0 else TaskStatus.FAILED)
            # Resolve the Future — unblocks the backend thread.
            if not task.completion_future.done():
                task.completion_future.set_result((exit_code, stdout, stderr))
        logger.info('Task %s for %s completed (exit_code=%d)', task_id,
                    cluster_name, exit_code)
        return True

    # ------------------------------------------------------------------ #
    # Heartbeat
    # ------------------------------------------------------------------ #

    def record_heartbeat(self, cluster_name: str) -> None:
        """Record a heartbeat from the poll worker."""
        with self._lock:
            worker = self._workers.get(cluster_name)
            if worker is None:
                worker = WorkerState(cluster_name=cluster_name)
                self._workers[cluster_name] = worker
                logger.info('Poll worker connected: %s', cluster_name)
            worker.last_heartbeat = time.time()

    def is_worker_online(
        self,
        cluster_name: str,
        timeout: float = HEARTBEAT_TIMEOUT_SECONDS,
    ) -> bool:
        """Check if the poll worker has heartbeated recently."""
        with self._lock:
            worker = self._workers.get(cluster_name)
        if worker is None:
            return False
        return (time.time() - worker.last_heartbeat) < timeout

    def get_last_heartbeat(self, cluster_name: str) -> Optional[float]:
        """Return the timestamp of the last heartbeat, or None."""
        with self._lock:
            worker = self._workers.get(cluster_name)
        if worker is None:
            return None
        return worker.last_heartbeat

    # ------------------------------------------------------------------ #
    # Ephemeral key file tracking
    # ------------------------------------------------------------------ #

    def register_ephemeral_key_files(self, cluster_name: str,
                                     paths: List[str]) -> None:
        """Track temp key files for cleanup on cluster teardown."""
        with self._lock:
            self._ephemeral_key_files[cluster_name] = [
                p for p in paths if p and p.startswith('/tmp/')
            ]

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def cleanup_cluster(self, cluster_name: str) -> None:
        """Remove all state for a cluster (on teardown)."""
        with self._lock:
            # Cancel any pending futures so blocked threads unblock.
            for task in self._tasks.get(cluster_name, []):
                if not task.completion_future.done():
                    task.completion_future.cancel()
            self._tasks.pop(cluster_name, None)
            self._workers.pop(cluster_name, None)
            self._signing_keys.pop(cluster_name, None)
            # Delete ephemeral SSH key temp files.
            for path in self._ephemeral_key_files.pop(cluster_name, []):
                try:
                    os.unlink(path)
                    logger.debug('Deleted ephemeral key file %s', path)
                except OSError:
                    pass
        self.remove_tokens(cluster_name)
        logger.info('Cleaned up state for cluster %s', cluster_name)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


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
        """Poll worker fetches the next pending task.

        Returns the task payload with an Ed25519 signature (covering the
        worker-supplied nonce for replay protection), or {"task": null}
        if no tasks are queued.
        """
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


# ---------------------------------------------------------------------- #
# Ephemeral credential store
#
# Credentials are stashed here between request receipt and provisioning.
# They live only in server memory — never persisted to disk or database.
# The provisioner pops them (deleting from memory) and writes to temp
# files that are cleaned up after the SSH operation completes.
# ---------------------------------------------------------------------- #

# Keyed by (user_hash, cluster_name) so different users can't access
# each other's credentials even if they share a cluster name.
_ephemeral_credentials: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
_cred_lock = threading.Lock()


def stash_credentials(user_hash: str,
                      cluster_name: str,
                      private_key_content: str,
                      certificate_content: Optional[str] = None,
                      ssh_user: Optional[str] = None) -> None:
    """Stash credentials in memory for the provisioner to retrieve."""
    key = (user_hash, cluster_name)
    with _cred_lock:
        _ephemeral_credentials[key] = {
            'private_key_content': private_key_content,
            'certificate_content': certificate_content,
            'ssh_user': ssh_user,
        }
    logger.info('Stashed ephemeral credentials for user=%s cluster=%s',
                user_hash[:8], cluster_name)


def peek_credentials(
    user_hash: str,
    cluster_name: str,
) -> Optional[Dict[str, Optional[str]]]:
    """Read credentials without removing them. Returns None if not stashed."""
    key = (user_hash, cluster_name)
    with _cred_lock:
        return _ephemeral_credentials.get(key)


def pop_credentials(
    user_hash: str,
    cluster_name: str,
) -> Optional[Dict[str, Optional[str]]]:
    """Pop credentials from the store. Returns None if not stashed."""
    key = (user_hash, cluster_name)
    with _cred_lock:
        return _ephemeral_credentials.pop(key, None)
