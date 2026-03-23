"""Slurm adaptor for SkyPilot."""

import ipaddress
import logging
import os
import re
import shlex
import sqlite3
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from sky.adaptors import common
from sky.skylet import runtime_utils
from sky.utils import command_runner
from sky.utils import subprocess_utils
from sky.utils import timeline

logger = logging.getLogger(__name__)

# ASCII Unit Separator (\x1f) to handle values with spaces
# and other special characters.
SEP = r'\x1f'

# Regex pattern to extract partition names from scontrol output
# Matches PartitionName=<name> and captures until the next field
_PARTITION_NAME_REGEX = re.compile(r'PartitionName=(.+?)(?:\s+\w+=|$)')

# Regex pattern to extract MAXTIME from scontrol output
# Matches MaxTime=<time> and captures the time
_MAXTIME_REGEX = re.compile(r'MaxTime=((?:\d+-)?\d{1,2}:\d{2}:\d{2}|UNLIMITED)')

_IMPORT_ERROR_MESSAGE = ('Failed to import dependencies for Slurm. '
                         'Try running: pip install "skypilot[slurm]"')
hostlist = common.LazyImport('hostlist',
                             import_error_message=_IMPORT_ERROR_MESSAGE)

_UNRESOLVED_HOSTNAME_MARKER = 'UNRESOLVED'

# Default poll interval for SlurmJobInfo freshness checks (seconds).
_DEFAULT_POLL_INTERVAL = 10


class SlurmPartition(NamedTuple):
    """Information about the Slurm partitions."""
    name: str
    is_default: bool
    # The maximum time a job can run in seconds.
    # None if the maximum time is unlimited.
    maxtime: Optional[int]


# TODO(kevin): Add more API types for other client functions.
class NodeInfo(NamedTuple):
    """Information about a Slurm node from sinfo."""
    node: str
    state: str
    gres: str
    cpus: int
    memory_gb: float
    # The default partition contains a '*' at the end of the name.
    # It is the caller's responsibility to strip the '*' if needed.
    partition: str


class JobInfo(NamedTuple):
    """Job information from squeue."""
    job_id: str
    state: Optional[str]  # e.g., 'RUNNING', 'PENDING', or None if not found
    reason: Optional[str]  # e.g., 'Priority', 'Resources', or None
    nodelist: Optional[str]  # e.g., 'node[01-03]', or None if no nodes


def _parse_maxtime(line: str) -> Optional[int]:
    """Parse the maximum time a job can run from the scontrol output."""
    maxtime_match = _MAXTIME_REGEX.search(line)
    if not maxtime_match:
        return None
    maxtime_str = maxtime_match.group(1).strip()
    if maxtime_str == 'UNLIMITED':
        return None

    # Convert maxTime from '[days-]hours:minutes:seconds' to seconds.
    # Example: "2-12:30:05" => (2*86400) + (12*3600) + (30*60) + 5
    days = 0
    time_part = maxtime_str
    if '-' in maxtime_str:
        days_part, time_part = maxtime_str.split('-', 1)
        days = int(days_part)

    h, m, s = map(int, time_part.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s


class SlurmClient:
    """Client for Slurm control plane operations."""

    def __init__(
        self,
        ssh_host: Optional[str] = None,
        ssh_port: Optional[int] = None,
        ssh_user: Optional[str] = None,
        ssh_key: Optional[str] = None,
        ssh_proxy_command: Optional[str] = None,
        ssh_proxy_jump: Optional[str] = None,
        is_inside_slurm_cluster: bool = False,
        identities_only: Optional[bool] = None,
    ):
        """Initialize SlurmClient.

        Args:
            ssh_host: Hostname of the Slurm controller.
            ssh_port: SSH port on the controller.
            ssh_user: SSH username.
            ssh_key: Path to SSH private key, or None for keyless SSH.
            ssh_proxy_command: Optional SSH proxy command.
            ssh_proxy_jump: Optional SSH proxy jump destination.
            is_inside_slurm_cluster: If True, uses local execution mode (for
            when running on the Slurm cluster itself). Defaults to False.
            identities_only: If True, only use the specified identity file and
                don't try ssh-agent keys. If None, defaults to False (allows
                ssh-agent fallback for backward compatibility).
        """
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.ssh_proxy_command = ssh_proxy_command
        self.ssh_proxy_jump = ssh_proxy_jump

        self._runner: command_runner.CommandRunner

        if is_inside_slurm_cluster:
            # Local execution mode - for running on the Slurm cluster itself
            # (e.g., autodown from skylet).
            self._runner = command_runner.LocalProcessCommandRunner()
        else:
            # Remote execution via SSH
            assert ssh_host is not None
            assert ssh_port is not None
            assert ssh_user is not None
            # If user has IdentitiesOnly=yes in their config, respect it by
            # NOT disabling IdentitiesOnly. Otherwise, allow ssh-agent fallback.
            self._runner = command_runner.SSHCommandRunner(
                (ssh_host, ssh_port),
                ssh_user,
                ssh_key,
                ssh_proxy_command=ssh_proxy_command,
                ssh_proxy_jump=ssh_proxy_jump,
                enable_interactive_auth=True,
                disable_identities_only=not identities_only,
            )

    def _run_slurm_cmd(self, cmd: str) -> Tuple[int, str, str]:
        return self._runner.run(cmd,
                                require_outputs=True,
                                separate_stderr=True,
                                stream_logs=False)

    def cancel_jobs_by_name(self,
                            job_name: str,
                            signal: Optional[str] = None,
                            full: bool = False) -> None:
        """Cancel Slurm job(s) by name.

        Args:
            job_name: Name of the job(s) to cancel.
            signal: Optional signal to send to the job(s).
            full: If True, signals the batch script and its children processes.
                By default, signals other than SIGKILL are not sent to the
                batch step (the shell script).
        """
        cmd = f'scancel --name {job_name}'
        if signal is not None:
            cmd += f' --signal {signal}'
        if full:
            cmd += ' --full'
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           f'Failed to cancel job {job_name}.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)
        logger.debug(f'Successfully cancelled job {job_name}: {stdout}')

    def info(self) -> str:
        """Get Slurm cluster information.

        This is useful for checking if the cluster is accessible and
        retrieving node information.

        Returns:
            The stdout output from sinfo.
        """
        cmd = 'sinfo'
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            'Failed to get Slurm cluster information.',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)
        return stdout

    def info_nodes(self) -> List[NodeInfo]:
        """Get Slurm node information.

        Returns node names, states, GRES (generic resources like GPUs),
        CPUs, memory (MB), and partitions.
        """
        cmd = (f'sinfo -h --Node -o '
               f'"%N{SEP}%t{SEP}%G{SEP}%c{SEP}%m{SEP}%P"')
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            'Failed to get Slurm node information.',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)

        nodes = []
        for line in stdout.splitlines():
            parts = line.split(SEP)
            if len(parts) != 6:
                raise RuntimeError(
                    f'Unexpected output format from sinfo: {line!r}')
            try:
                node_info = NodeInfo(node=parts[0],
                                     state=parts[1],
                                     gres=parts[2],
                                     cpus=int(parts[3]),
                                     memory_gb=int(parts[4]) / 1024.0,
                                     partition=parts[5])
                nodes.append(node_info)
            except ValueError as e:
                raise RuntimeError(
                    f'Failed to parse node info from line: {line!r}. '
                    f'Error: {e}') from e

        return nodes

    def node_details(self, node_name: str) -> Dict[str, str]:
        """Get detailed Slurm node information.

        Returns:
            A dictionary of node attributes.
        """

        def _parse_scontrol_node_output(output: str) -> Dict[str, str]:
            """Parses the key=value output of 'scontrol show node'."""
            node_info = {}
            # Split by space, handling values that might have spaces
            # if quoted. This is simplified; scontrol can be complex.
            parts = output.split()
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    # Simple quote removal, might need refinement
                    value = value.strip('\'"')
                    node_info[key] = value
            return node_info

        cmd = f'scontrol show node {node_name}'
        rc, node_details, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            f'Failed to get detailed node information for {node_name}.',
            stderr=f'{node_details}\n{stderr}',
            stream_logs=False)
        node_info = _parse_scontrol_node_output(node_details)
        return node_info

    def get_all_jobs_gres(self) -> Dict[str, List[str]]:
        """Get GRES allocation for all running jobs, grouped by node.

        Returns:
            Dict mapping node_name -> list of GRES strings for jobs on that
            node.
        """
        cmd = f'squeue -h --states=running,completing -o "%N{SEP}%b"'
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           'Failed to get all jobs GRES.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)

        nodes_to_gres: Dict[str, List[str]] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(SEP)
            if len(parts) != 2:
                # We should never reach here, but just in case.
                continue
            nodelist_str, gres_str = parts
            if not gres_str or gres_str == 'N/A':
                continue

            for node in hostlist.expand_hostlist(nodelist_str):
                nodes_to_gres.setdefault(node, []).append(gres_str)

        return nodes_to_gres

    def get_pending_job_count(self,
                              partition: str,
                              exclude_job_id: Optional[str] = None) -> int:
        """Count pending jobs in a partition, excluding our own job.

        Args:
            partition: The Slurm partition to query.
            exclude_job_id: Optional job ID to exclude from the count.

        Returns:
            The number of pending jobs, or -1 if the query fails.
        """
        cmd = f'squeue -h -p {partition} --states=pending -o "%i"'
        rc, stdout, _ = self._run_slurm_cmd(cmd)
        if rc != 0:
            return -1
        job_ids = [j.strip() for j in stdout.strip().splitlines() if j.strip()]
        if exclude_job_id:
            job_ids = [j for j in job_ids if j != exclude_job_id]
        return len(job_ids)

    def query_jobs(
        self,
        job_name: Optional[str] = None,
        state_filters: Optional[List[str]] = None,
    ) -> List[JobInfo]:
        """Query jobs, returning state, reason, and nodelist per job.

        A single squeue call that returns all requested job information.
        Typically called via SlurmJobInfo.poll() rather than directly.

        Args:
            job_name: Optional job name to filter by (--name).
            state_filters: List of job states to filter by
                (e.g., ['all'] for all states). If None, uses squeue
                defaults (active jobs only).

        Returns:
            List of JobInfo with job_id, state, reason, and nodelist.
        """
        cmd = f'squeue --me -h -o "%i{SEP}%T{SEP}%r{SEP}%N"'
        if state_filters is not None:
            state_filters_str = ','.join(state_filters)
            cmd += f' --states {state_filters_str}'
        if job_name is not None:
            cmd += f' --name {job_name}'

        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           'Failed to query Slurm jobs.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)

        results: List[JobInfo] = []
        for line in stdout.splitlines():
            # Split on \x1f before stripping, since strip() removes \x1f.
            parts = line.split('\x1f')
            if len(parts) != 4:
                if line.strip():
                    logger.debug(f'Unexpected squeue output line: {line!r}')
                continue
            parsed_id, state, reason, nodelist = parts
            reason = reason.strip()
            reason = reason if reason and reason != 'None' else None
            nodelist = nodelist.strip() or None
            results.append(
                JobInfo(job_id=parsed_id.strip(),
                        state=state.strip(),
                        reason=reason,
                        nodelist=nodelist))
        return results

    def job_tracker(
        self,
        cluster_name: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> 'SlurmJobInfo':
        """Get a SlurmJobInfo tracker for the given cluster.

        Args:
            cluster_name: The cluster name (used as squeue --name filter
                and SQLite cache key).
            poll_interval: Minimum seconds between squeue calls.

        Returns:
            A SlurmJobInfo instance backed by a shared SQLite cache.
        """
        return SlurmJobInfo(self, cluster_name, poll_interval)

    @timeline.event
    def get_job_nodes(self, job_id: str) -> Tuple[List[str], List[str]]:
        """Get the list of nodes and their IPs for a given job ID.

        The ordering is guaranteed to be stable for the lifetime of the job.

        Args:
            job_id: The Slurm job ID.

        Returns:
            A tuple of (nodes, node_ips) where nodes is a list of node names
            and node_ips is a list of corresponding IP addresses.
        """

        cmd = (
            # Use scontrol show hostnames to expand both compact Slurm
            # hostlist notation (e.g. ml-16-node-[001-002]) and
            # comma-separated nodes into individual node names.
            # TODO(kevin): Use json output for more robust parsing.
            f'nodelist=$(squeue -h --jobs {job_id} -o "%N"); '
            f'scontrol show hostnames $nodelist | while read -r node; do '
            f'node_addr=$(scontrol show node=$node | grep NodeAddr= | '
            f'awk -F= \'{{print $2}}\' | awk \'{{print $1}}\'); '
            f'echo "$node $node_addr"; '
            f'done')
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            f'Failed to get nodes for job {job_id}.',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)
        logger.debug(f'Successfully got nodes for job {job_id}: {stdout}')

        node_info = {}
        nodes_to_resolve: List[Tuple[str, str]] = []

        for line in stdout.strip().splitlines():
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    node_name = parts[0]
                    node_addr = parts[1]
                    try:
                        ipaddress.ip_address(node_addr)
                        node_info[node_name] = node_addr  # Already an IP
                    except ValueError:
                        nodes_to_resolve.append((node_name, node_addr))

        if nodes_to_resolve:
            hostnames = [h for _, h in nodes_to_resolve]
            # The output of `getent ahostsv4` is as follows:
            # 10.0.0.0     STREAM worker-0
            # 10.0.0.0     DGRAM
            # 10.0.0.0     RAW
            resolve_ip_cmd = (
                f'for h in {" ".join(hostnames)}; do '
                f'ip=$(getent ahostsv4 "$h" | head -1 | awk \'{{print $1}}\'); '
                f'if [ -n "$ip" ]; then echo "$h $ip"; '
                f'else echo "$h {_UNRESOLVED_HOSTNAME_MARKER}"; fi; '
                f'done')
            rc, resolve_stdout, stderr = self._run_slurm_cmd(resolve_ip_cmd)
            subprocess_utils.handle_returncode(
                rc,
                resolve_ip_cmd,
                f'Failed to resolve hostnames for: {hostnames}',
                stderr=f'{resolve_stdout}\n{stderr}',
                stream_logs=False)

            hostname_to_ip = {}
            unresolved = []
            for line in resolve_stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    hostname = parts[0]
                    ip = parts[1]
                    if ip == _UNRESOLVED_HOSTNAME_MARKER:
                        unresolved.append(hostname)
                    else:
                        hostname_to_ip[hostname] = ip

            if unresolved:
                raise RuntimeError(
                    f'Failed to resolve hostnames for: {unresolved}')

            for node_name, hostname in nodes_to_resolve:
                if hostname not in hostname_to_ip:
                    raise RuntimeError(
                        f'Failed to resolve {hostname} for node {node_name}')
                node_info[node_name] = hostname_to_ip[hostname]

        nodes = list(node_info.keys())
        node_ips = [node_info[node] for node in nodes]
        if not nodes:
            raise RuntimeError(
                f'No nodes found for job {job_id}. '
                f'The job may have terminated or the output was empty.')
        assert (len(nodes) == len(node_ips)
               ), f'Number of nodes and IPs do not match: {nodes} != {node_ips}'

        return nodes, node_ips

    def submit_job(
        self,
        partition: str,
        job_name: str,
        script_path: str,
    ) -> str:
        """Submit a Slurm job script.

        Args:
            partition: Slurm partition to submit to.
            job_name: Name to give the job.
            script_path: Remote path where the script will be stored.

        Returns:
            The job ID of the submitted job.
        """
        cmd = f'sbatch --partition={partition} {script_path}'
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           'Failed to submit Slurm job.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)

        # Parse job ID from sbatch output (format: "Submitted batch job 12345")
        job_id_match = re.search(r'Submitted batch job (\d+)', stdout)
        if not job_id_match:
            raise RuntimeError(
                f'Failed to parse job ID from sbatch output: {stdout}')

        job_id = job_id_match.group(1).strip()
        logger.debug(f'Successfully submitted Slurm job {job_id} with name '
                     f'{job_name}: {stdout}')

        return job_id

    def get_partitions_info(self) -> List[SlurmPartition]:
        """Get the partitions information for the Slurm cluster.

        Returns:
            List of SlurmPartition objects.
        """
        cmd = 'scontrol show partitions -o'
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           'Failed to get Slurm partitions.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)

        partitions = []
        for line in stdout.strip().splitlines():
            is_default = False
            match = _PARTITION_NAME_REGEX.search(line)
            if 'Default=YES' in line:
                is_default = True
            maxtime = _parse_maxtime(line)
            if match:
                partition = match.group(1).strip()
                if partition:
                    partitions.append(
                        SlurmPartition(name=partition,
                                       is_default=is_default,
                                       maxtime=maxtime))
        return partitions

    def get_default_partition(self) -> Optional[str]:
        """Get the default partition name for the Slurm cluster.

        Returns:
            The default partition name, or None if it cannot be determined.
        """
        partitions = self.get_partitions_info()
        for partition in partitions:
            if partition.is_default:
                return partition.name
        return None

    def get_partitions(self) -> List[str]:
        """Get unique partition names in the Slurm cluster.

        Returns:
            List of partition names. The default partition will not have a '*'
            at the end of the name.
        """
        return [partition.name for partition in self.get_partitions_info()]

    def get_proctrack_type(self) -> Optional[str]:
        """Get the ProctrackType from Slurm configuration.

        Returns:
            The proctrack type (e.g., 'cgroup', 'linuxproc', 'pgid'),
            or None if it cannot be determined.
        """
        cmd = 'scontrol show config | grep -i "^ProctrackType"'
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        if rc != 0:
            logger.warning(f'Failed to get ProctrackType: {stderr}')
            return None

        # Parse output like "ProctrackType           = proctrack/cgroup"
        match = re.search(r'ProctrackType\s*=\s*proctrack/(\w+)', stdout)
        if match:
            return match.group(1)
        return None

    def check_pyxis_enabled(self) -> bool:
        """Check if the Pyxis SPANK plugin is installed.

        Pyxis registers --container-* flags tagged with [pyxis] in srun
        help output. This is a reliable way to detect the plugin without
        requiring a job allocation.

        Returns:
            True if Pyxis is installed, False otherwise.
        """
        cmd = 'srun --help 2>&1 | grep -q \'\\[pyxis\\]\''
        rc, _, _ = self._run_slurm_cmd(cmd)
        return rc == 0

    def get_env(self) -> Dict[str, str]:
        """Fetch environment variables from the remote host.

        Returns:
            Dictionary of environment variable name -> value.
        """
        rc, stdout, stderr = self._run_slurm_cmd('env')
        if rc != 0:
            logger.warning(f'Failed to fetch remote env: {stderr}')
            return {}
        env: Dict[str, str] = {}
        for line in stdout.splitlines():
            if '=' in line:
                key, _, value = line.partition('=')
                env[key] = value
        return env

    def get_remote_home_dir(self) -> str:
        """Returns the remote user's home directory."""
        return self._runner.get_remote_home_dir()

    def check_file_exists(self, path: str) -> bool:
        """Check if a file exists on the remote host."""
        cmd = f'test -f {shlex.quote(path)}'
        rc, stdout, stderr = self._run_slurm_cmd(cmd)
        if rc not in (0, 1):
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to check for file: {path}',
                stderr=f'{stdout}\n{stderr}')
        return rc == 0

    def check_fuse_enabled(self) -> bool:
        """Check if FUSE is available on the cluster.

        FUSE is required for mounting object stores (e.g., via goofys or
        rclone). We check for /dev/fuse which is the device node that FUSE
        requires.

        We first try to check on a compute node via srun, since that is
        where mounts actually happen. If srun cannot allocate resources
        (cluster is full, etc.), we fall back to checking the login node.

        Returns:
            True if FUSE is available, False otherwise.
        """
        # Try checking on a compute node first. We use a wrapper that
        # prints a marker so we can distinguish "command ran and /dev/fuse
        # is missing" from "srun itself failed to allocate".
        srun_cmd = ('srun --immediate=10 --time=00:00:30 '
                    'bash -c \'test -e /dev/fuse '
                    '&& echo FUSE_OK || echo FUSE_MISSING\'')
        rc, stdout, _ = self._run_slurm_cmd(srun_cmd)
        stdout = stdout.strip()
        if rc == 0 and 'FUSE_OK' in stdout:
            return True
        if rc == 0 and 'FUSE_MISSING' in stdout:
            return False

        # srun failed (no resources, misconfigured, etc.).
        # Fall back to checking the login node.
        logger.debug('srun FUSE check failed, falling back to login node')
        cmd = 'test -e /dev/fuse'
        rc, _, _ = self._run_slurm_cmd(cmd)
        return rc == 0

    def check_dir_shared_fs(self, path: str) -> Optional[str]:
        """Check the filesystem type of a directory.

        Args:
            path: The directory path to check. Must be an absolute path
                (no shell variables or ~).

        Returns:
            The filesystem type string (e.g., 'nfs', 'ext2/ext3'),
            or None if the check could not be performed.
        """
        cmd = f'stat -f -c %T {shlex.quote(path)}'
        rc, stdout, _ = self._run_slurm_cmd(cmd)
        if rc != 0:
            return None
        return stdout.strip().lower()

    def check_homedir_shared_fs(self) -> Optional[str]:
        """Check the filesystem type of the home directory."""
        return self.check_dir_shared_fs('~')


# SQLite timeout for write lock contention (seconds).
_DB_TIMEOUT_S = 60


class SlurmJobInfo:
    """SQLite-backed cache of Slurm job state.

    Provides a single-poll, multi-read interface for squeue data.
    The SQLite DB is file-backed and shared across API server worker
    processes, so only one process needs to call squeue per poll
    interval.

    Usage:
        tracker = SlurmJobInfo(client, 'my-cluster', poll_interval=10)
        tracker.poll()            # Runs squeue if cache is stale
        info = tracker.job_info('12345')  # Read from cache
        ids = tracker.jobs()              # All cached job IDs
    """

    @classmethod
    def _get_db_path(cls) -> str:
        return runtime_utils.get_runtime_dir_path('.sky/slurm_job_cache.db')

    def __init__(self,
                 client: 'SlurmClient',
                 cluster_name: str,
                 poll_interval: float = _DEFAULT_POLL_INTERVAL):
        self._client = client
        self._cluster_name = cluster_name
        self._poll_interval = poll_interval
        self._conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        db_path = self._get_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=_DB_TIMEOUT_S)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute("""
            CREATE TABLE IF NOT EXISTS slurm_jobs (
                cluster_name TEXT NOT NULL,
                job_id TEXT NOT NULL,
                state TEXT,
                reason TEXT,
                nodelist TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (cluster_name, job_id)
            )
        """)
        conn.commit()
        return conn

    def poll(self) -> None:
        """Refresh job state from squeue if the cache is stale.

        Checks the most recent updated_at for this cluster. If the
        data is younger than poll_interval, the squeue call is skipped
        (cross-process deduplication).
        """
        row = self._conn.execute(
            'SELECT MAX(updated_at) FROM slurm_jobs '
            'WHERE cluster_name = ?',
            (self._cluster_name,),
        ).fetchone()
        if row[0] is not None and time.time() - row[0] < self._poll_interval:
            return  # Cache is fresh.

        results = self._client.query_jobs(job_name=self._cluster_name,
                                          state_filters=['all'])
        now = time.time()
        self._conn.execute(
            'DELETE FROM slurm_jobs WHERE cluster_name = ?',
            (self._cluster_name,),
        )
        self._conn.executemany(
            'INSERT INTO slurm_jobs '
            '(cluster_name, job_id, state, reason, nodelist, updated_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [(self._cluster_name, j.job_id, j.state, j.reason, j.nodelist, now)
             for j in results],
        )
        self._conn.commit()

    def job_info(self, job_id: str) -> Optional[JobInfo]:
        """Get cached info for a specific job."""
        row = self._conn.execute(
            'SELECT job_id, state, reason, nodelist FROM slurm_jobs '
            'WHERE cluster_name = ? AND job_id = ?',
            (self._cluster_name, job_id),
        ).fetchone()
        if row is None:
            return None
        return JobInfo(job_id=row[0],
                       state=row[1],
                       reason=row[2],
                       nodelist=row[3])

    def jobs(self) -> List[str]:
        """Return all cached job IDs for this cluster."""
        rows = self._conn.execute(
            'SELECT job_id FROM slurm_jobs WHERE cluster_name = ?',
            (self._cluster_name,),
        ).fetchall()
        return [r[0] for r in rows]
