"""Regression test: sky queue must report the actual log directory
(job-ID-based, e.g. ~/sky_logs/3-sky-cmd/) rather than the
timestamp-based path (e.g. ~/sky_logs/sky-2026-03-31-...).

The Skylet's add_job() creates the log directory as
    ~/sky_logs/{job_id}-{job_name}
and persists it in the log_dir column of the jobs table. But
get_job_queue (both legacy and gRPC paths) was constructing log_path
from run_timestamp, ignoring log_dir entirely. This caused sky logs
to look for a nonexistent directory and hang.
"""
import subprocess
import sys


def test_job_queue_returns_log_dir_not_run_timestamp():
    """The log_path returned by dump_job_queue must match the log_dir
    set by add_job(), not the run_timestamp-based path.

    add_job() creates ~/sky_logs/{job_id}-{job_name} and persists it
    in the DB. dump_job_queue must return this path, not
    ~/sky_logs/{run_timestamp}.

    Runs in a subprocess to avoid mutating module-level DB state.
    """
    result = subprocess.run(
        [
            sys.executable, '-c', """
import json
import os
import tempfile

# Use a temporary DB so we don't affect the real job queue.
tmpdir = tempfile.mkdtemp()
os.environ['SKYPILOT_JOB_DB_PATH'] = os.path.join(tmpdir, 'jobs.db')

from sky.skylet import constants
from sky.skylet import job_lib
from sky.utils import message_utils

job_name = 'test-job'
run_timestamp = 'sky-2026-03-31-12-00-00-000000'

job_id, log_dir = job_lib.add_job(
    job_name=job_name,
    username='test-user',
    run_timestamp=run_timestamp,
    resources_str='1x[CPU:1]',
)

# The actual log_dir should be job-ID-based.
expected = os.path.join(constants.SKY_LOGS_DIRECTORY, f'{job_id}-{job_name}')
assert log_dir == expected, f'add_job: {log_dir} != {expected}'

# Now get the job queue and check log_path.
queue_payload = job_lib.dump_job_queue(user_hash=None, all_jobs=True)
jobs = message_utils.decode_payload(queue_payload)
job = next(j for j in jobs if j['job_id'] == job_id)

actual = job['log_path']
if actual != expected:
    print(f'FAIL: log_path={actual}, expected={expected}')
    raise SystemExit(1)
print('OK')
"""
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'dump_job_queue log_path must come from log_dir DB column, '
        f'not run_timestamp:\n{result.stdout.strip()}\n'
        f'{result.stderr.strip()[-500:]}')
