"""Tests for SSH resilience in the Slurm podman-hpc backend.

Verifies that transient SSH failures don't crash the managed jobs
controller or propagate as fatal errors. The key invariant: for
podman-hpc, SSH is a convenience layer (log tailing), not the
execution layer (task queue). SSH failures should be retried or
degraded gracefully, never fatal.
"""
import threading
from unittest import mock

import pytest

from sky import exceptions
from sky.provision.slurm import instance as slurm_instance


class TestCommandRunnerCache:
    """Verify get_command_runners() caches SSH-derived values."""

    def setup_method(self):
        """Clear the cache before each test."""
        with slurm_instance._command_runner_cache_lock:
            slurm_instance._command_runner_cache.clear()

    def _make_cluster_info(self, cluster_name='test-cluster'):
        """Build a minimal ClusterInfo for get_command_runners()."""
        from sky.provision import common
        instance_info = common.InstanceInfo(
            instance_id='job-123',
            internal_ip='10.0.0.1',
            external_ip='login.example.com',
            ssh_port=22,
            tags={
                'skypilot-cluster-name': cluster_name,
                'job_id': '123',
                'node': 'node01',
            },
        )
        return common.ClusterInfo(
            instances={'job-123': [instance_info]},
            head_instance_id='job-123',
            provider_name='slurm',
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': 22,
                    'user': 'testuser',
                    'private_key': '/tmp/fake_key',
                },
                'cluster': 'test-slurm',
                'container_runtime': 'podman-hpc',
            },
        )

    @mock.patch('sky.utils.command_runner.SlurmCommandRunner')
    @mock.patch('sky.adaptors.slurm.SlurmClient')
    def test_second_call_uses_cache(self, mock_slurm_client_cls,
                                    mock_runner_cls):
        """Second call to get_command_runners() should not SSH."""
        mock_client = mock.MagicMock()
        mock_client.get_remote_home_dir.return_value = '/home/testuser'
        mock_client.check_file_exists.return_value = True
        mock_slurm_client_cls.return_value = mock_client

        info = self._make_cluster_info()

        # First call — SSHes to get home dir and check container marker
        runners1 = slurm_instance.get_command_runners(info)
        assert len(runners1) == 1
        assert mock_client.get_remote_home_dir.call_count == 1
        assert mock_client.check_file_exists.call_count == 1

        # Second call — should use cache, no new SSH calls
        runners2 = slurm_instance.get_command_runners(info)
        assert len(runners2) == 1
        assert mock_client.get_remote_home_dir.call_count == 1  # unchanged
        assert mock_client.check_file_exists.call_count == 1  # unchanged

    @mock.patch('sky.utils.command_runner.SlurmCommandRunner')
    @mock.patch('sky.adaptors.slurm.SlurmClient')
    def test_invalidate_cache_forces_ssh(self, mock_slurm_client_cls,
                                         mock_runner_cls):
        """invalidate_command_runner_cache() forces a fresh SSH call."""
        mock_client = mock.MagicMock()
        mock_client.get_remote_home_dir.return_value = '/home/testuser'
        mock_client.check_file_exists.return_value = True
        mock_slurm_client_cls.return_value = mock_client

        info = self._make_cluster_info()

        # Populate cache
        slurm_instance.get_command_runners(info)
        assert mock_client.get_remote_home_dir.call_count == 1

        # Invalidate
        slurm_instance.invalidate_command_runner_cache('test-cluster')

        # Next call must SSH again
        slurm_instance.get_command_runners(info)
        assert mock_client.get_remote_home_dir.call_count == 2

    @mock.patch('sky.utils.command_runner.SlurmCommandRunner')
    @mock.patch('sky.adaptors.slurm.SlurmClient')
    def test_different_clusters_independent(self, mock_slurm_client_cls,
                                            mock_runner_cls):
        """Different cluster names have independent cache entries."""
        mock_client = mock.MagicMock()
        mock_client.get_remote_home_dir.return_value = '/home/testuser'
        mock_client.check_file_exists.return_value = True
        mock_slurm_client_cls.return_value = mock_client

        info_a = self._make_cluster_info('cluster-a')
        info_b = self._make_cluster_info('cluster-b')

        slurm_instance.get_command_runners(info_a)
        assert mock_client.get_remote_home_dir.call_count == 1

        slurm_instance.get_command_runners(info_b)
        assert mock_client.get_remote_home_dir.call_count == 2

        # Invalidating A doesn't affect B
        slurm_instance.invalidate_command_runner_cache('cluster-a')
        slurm_instance.get_command_runners(info_b)
        assert mock_client.get_remote_home_dir.call_count == 2  # still cached

    @mock.patch('sky.adaptors.slurm.SlurmClient')
    def test_ssh_failure_propagates_on_first_call(self, mock_slurm_client_cls):
        """If SSH fails on first call (no cache), error propagates."""
        mock_client = mock.MagicMock()
        mock_client.get_remote_home_dir.side_effect = ValueError(
            'kex_exchange_identification: Connection reset by peer')
        mock_slurm_client_cls.return_value = mock_client

        info = self._make_cluster_info()
        with pytest.raises(ValueError, match='kex_exchange_identification'):
            slurm_instance.get_command_runners(info)


class TestSSHErrorException:
    """Verify SSHError is properly defined and usable."""

    def test_ssh_error_is_exception(self):
        err = exceptions.SSHError('connection reset')
        assert isinstance(err, Exception)
        assert 'connection reset' in str(err)

    def test_ssh_error_from_command_error(self):
        """SSHError can wrap a CommandError."""
        cmd_err = exceptions.CommandError(returncode=255,
                                          command='ssh user@host',
                                          error_msg='Connection refused',
                                          detailed_reason='stderr output')
        msg = f'SSH failed: {cmd_err}'
        ssh_err = exceptions.SSHError(msg)
        ssh_err.__cause__ = cmd_err
        assert ssh_err.__cause__ is cmd_err


class TestGetJobStatusTransientErrors:
    """Verify SSHError is treated as transient in get_job_status."""

    @pytest.mark.asyncio
    async def test_ssh_error_returns_transient(self):
        """SSHError from backend.get_job_status should be treated as
        transient, not fatal."""
        from sky.backends import cloud_vm_ray_backend
        from sky.jobs import utils as managed_job_utils

        mock_backend = mock.MagicMock()

        # Create a mock that passes isinstance checks
        mock_handle = mock.MagicMock(
            spec=cloud_vm_ray_backend.CloudVmRayResourceHandle)
        mock_handle.is_grpc_enabled_with_flag = False

        # Simulate run_on_head raising SSHError
        mock_backend.get_job_status.side_effect = exceptions.SSHError(
            'connection reset')

        with mock.patch(
                'sky.jobs.utils.global_user_state'
                '.get_handle_from_cluster_name',
                return_value=mock_handle):
            status, reason = await managed_job_utils.get_job_status(
                mock_backend, 'test-cluster', job_id=1)

        assert status is None
        assert reason is not None
        assert 'SSH error' in reason
