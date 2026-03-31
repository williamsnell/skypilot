"""Regression test: sky.provision.slurm.instance must not trigger a
circular import through sky.server.common → sky.data → sky.clouds.

The failure mode (seen on SkyServe controller):
    sky.clouds.__init__ starts
    → sky.clouds.aws → sky.provision → sky.provision.slurm.instance
    → sky.server.common → sky.data.__init__ → sky.data.storage
    → str(clouds.AWS())  # AttributeError: 'AWS' not in partially
                         # initialized sky.clouds

Root cause: on the SkyServe controller, SKYPILOT_USING_REMOTE_API_SERVER
is set, which short-circuits the usage_lib → server_common pre-loading
chain. Without that pre-load, sky.server.common is first imported via
our top-level import in sky.provision.slurm.instance — which happens
mid-sky.clouds initialization, triggering the circular import.

The fix: sky.provision.slurm.instance must not import sky.server.common
at module level. The import should be deferred to function scope.
"""
import os
import subprocess
import sys


def test_import_sky_clouds_no_circular_import():
    """Importing sky.clouds must not fail with a circular import through
    sky.provision.slurm.instance → sky.server.common → sky.data.storage.

    Reproduces the SkyServe controller environment by setting
    SKYPILOT_USING_REMOTE_API_SERVER=1, which prevents the usage_lib
    pre-loading of sky.server.common. Without pre-loading,
    sky.server.common is first reached mid-sky.clouds initialization,
    and if imported at module level in sky.provision.slurm.instance,
    the chain sky.server.common → sky.data.storage → clouds.AWS()
    fails because sky.clouds is only partially initialized.
    """
    env = os.environ.copy()
    env['SKYPILOT_USING_REMOTE_API_SERVER'] = '1'

    result = subprocess.run(
        [sys.executable, '-c', 'import sky.clouds'],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f'Circular import when importing sky.clouds with '
        f'SKYPILOT_USING_REMOTE_API_SERVER=1 '
        f'(simulates SkyServe controller environment):\n'
        f'{result.stderr.strip()[-1000:]}')
