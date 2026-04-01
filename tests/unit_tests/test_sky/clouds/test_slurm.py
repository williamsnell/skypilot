"""Tests for Slurm cloud implementation."""

import os
from pathlib import Path
from unittest.mock import patch
import unittest.mock as mock

import pytest

from sky import resources as resources_lib
from sky.adaptors import slurm
from sky.clouds import slurm as slurm_cloud
from sky.provision.slurm import instance as slurm_instance
from sky.provision.slurm import utils as slurm_utils


class TestCheckInstanceFits:
    """Test slurm_utils.check_instance_fits()."""

    node_4cpu_16gb = slurm.NodeInfo(node='node1',
                                    state='idle',
                                    gres='(null)',
                                    cpus=4,
                                    memory_gb=16,
                                    partition='dev*')

    node_2cpu_8gb = slurm.NodeInfo(node='node1',
                                   state='idle',
                                   gres='(null)',
                                   cpus=2,
                                   memory_gb=8,
                                   partition='dev*')

    node_gpu_a10g = slurm.NodeInfo(node='node1',
                                   state='idle',
                                   gres='gpu:a10g:8',
                                   cpus=192,
                                   memory_gb=786,
                                   partition='gpus')

    node_2cpu_8gb_cpus = slurm.NodeInfo(node='node2',
                                        state='idle',
                                        gres='(null)',
                                        cpus=2,
                                        memory_gb=8,
                                        partition='dev*')

    @pytest.mark.parametrize(
        'nodes,instance_type,partition,expected_fits,reason_contains',
        [
            # No accelerators - fits
            ([node_4cpu_16gb], '2CPU--8GB', None, True, None),
            # No accelerators - insufficient resources
            ([node_2cpu_8gb], '4CPU--16GB', None, False, 'Max found: 2 CPUs'),
            # GPU - fits
            ([node_gpu_a10g], '64CPU--256GB--a10g:4', None, True, None),
            # GPU - type not available
            ([node_gpu_a10g
             ], '64CPU--256GB--h100:4', None, False, 'No GPU nodes matching'),
            # Partition filtering with default partition (*) handling
            ([node_2cpu_8gb_cpus], '1CPU--4GB', 'dev', True, None),
            # Resource exists but in different partition
            ([node_2cpu_8gb_cpus
             ], '2CPU--8GB', 'different_partition', False, 'No nodes found'),
        ])
    @patch('sky.provision.slurm.utils.kv_cache.get_cache_entry',
           return_value=None)
    @patch('sky.provision.slurm.utils.get_cluster_default_partition')
    @patch('sky.provision.slurm.utils.slurm.SlurmClient')
    @patch('sky.provision.slurm.utils.SSHConfig.from_path')
    def test_check_instance_fits(self, mock_ssh_config, mock_slurm_client,
                                 mock_default_partition, mock_kv_cache, nodes,
                                 instance_type, partition, expected_fits,
                                 reason_contains):
        """Test various scenarios for instance fitting."""
        mock_default_partition.return_value = 'dev'
        mock_ssh_config_obj = mock.MagicMock()
        mock_ssh_config_obj.lookup.return_value = {
            'hostname': '10.0.0.1',
            'port': '22',
            'user': 'slurm',
            'identityfile': ['/home/user/.ssh/id_rsa'],
        }
        mock_ssh_config.return_value = mock_ssh_config_obj

        mock_slurm_client_instance = mock.MagicMock()
        mock_slurm_client_instance.info_nodes.return_value = nodes
        mock_slurm_client.return_value = mock_slurm_client_instance

        kwargs = {'cluster': 'hyperpod', 'instance_type': instance_type}
        if partition:
            kwargs['partition'] = partition

        fits, reason = slurm_utils.check_instance_fits(**kwargs)

        assert fits is expected_fits
        if reason_contains:
            assert reason_contains in reason
        else:
            assert reason is None


class TestTerminateInstances:
    """Test slurm_instance.terminate_instances()."""

    @pytest.mark.parametrize(
        'job_state,should_cancel',
        [
            # Terminal states - no action needed
            ('COMPLETED', False),
            ('CANCELLED', False),
            ('FAILED', False),
            ('TIMEOUT', False),
            ('NODE_FAIL', False),
            ('PREEMPTED', False),
            ('SPECIAL_EXIT', False),
            # COMPLETING - already terminating
            ('COMPLETING', False),
            # All other states - plain scancel (actually cancels the job)
            ('PENDING', True),
            ('CONFIGURING', True),
            ('RUNNING', True),
            ('SUSPENDED', True),
            ('STAGING_OUT', True),
        ])
    def test_terminate_instances_handles_job_states(self, tmp_path, job_state,
                                                    should_cancel):
        """Test terminate_instances handles different job states correctly.

        All cancellable states use plain scancel (no --signal flag).
        scancel --signal TERM only delivers the signal without actually
        cancelling the job, so the job stays RUNNING if processes survive.
        """
        from sky.adaptors.slurm import JobInfo
        from sky.adaptors.slurm import SlurmClient
        from sky.adaptors.slurm import SlurmJobInfo

        mock_client = mock.MagicMock()
        mock_client.job_tracker.side_effect = lambda name, **kw: (
            SlurmClient.job_tracker(mock_client, name, **kw))
        mock_client.query_jobs.return_value = [
            JobInfo(job_id='12345', state=job_state, reason=None,
                    nodelist=None),
        ]

        cluster_name = 'test-cluster'
        provider_config = {
            'ssh': {
                'hostname': 'localhost',
                'port': '22',
                'user': 'testuser',
                'private_key': '/path/to/key',
            }
        }

        with patch('sky.provision.slurm.instance.slurm.SlurmClient',
                   return_value=mock_client), \
             patch('sky.provision.slurm.instance.slurm_utils'
                   '.is_inside_slurm_cluster', return_value=False), \
             patch.object(SlurmJobInfo, '_get_db_path',
                          return_value=str(tmp_path / 'test.db')):
            slurm_instance.terminate_instances(
                cluster_name_on_cloud=cluster_name,
                provider_config=provider_config,
            )

        if should_cancel:
            mock_client.cancel_jobs_by_name.assert_called_once_with(
                cluster_name)
        else:
            mock_client.cancel_jobs_by_name.assert_not_called()

    def test_terminate_instances_no_jobs_found(self, tmp_path):
        """Test terminate_instances when no jobs are found."""
        from sky.adaptors.slurm import SlurmClient
        from sky.adaptors.slurm import SlurmJobInfo

        mock_client = mock.MagicMock()
        mock_client.job_tracker.side_effect = lambda name, **kw: (
            SlurmClient.job_tracker(mock_client, name, **kw))
        mock_client.query_jobs.return_value = []

        cluster_name = 'test-cluster'
        provider_config = {
            'ssh': {
                'hostname': 'localhost',
                'port': '22',
                'user': 'testuser',
                'private_key': '/path/to/key',
            }
        }

        with patch('sky.provision.slurm.instance.slurm.SlurmClient',
                   return_value=mock_client), \
             patch('sky.provision.slurm.instance.slurm_utils'
                   '.is_inside_slurm_cluster', return_value=False), \
             patch.object(SlurmJobInfo, '_get_db_path',
                          return_value=str(tmp_path / 'test.db')):
            slurm_instance.terminate_instances(
                cluster_name_on_cloud=cluster_name,
                provider_config=provider_config,
            )

        mock_client.cancel_jobs_by_name.assert_not_called()


class TestSlurmGPUDefaults:
    """Test Slurm GPU CPU/memory allocation.

    When GPU instances are requested without explicit CPU/memory, the
    instance type should be GPU-only (cpus=None, memory=None) so the
    Slurm scheduler can auto-allocate resources per GPU.

    When CPU and/or memory are explicitly specified, they should be
    included in the instance type.
    """

    @pytest.mark.parametrize('gpu_count', [1, 2, 4, 8])
    @patch('sky.clouds.slurm.Slurm.regions_with_offering')
    def test_gpu_without_explicit_cpu_memory_is_gpu_only(
            self, mock_regions, gpu_count):
        """GPU instances without explicit CPU/memory produce GPU-only type."""
        mock_region = mock.MagicMock()
        mock_region.name = 'test-cluster'
        mock_regions.return_value = [mock_region]

        resources = resources_lib.Resources(
            cloud=slurm_cloud.Slurm(),
            accelerators={'H200': gpu_count},
        )

        cloud = slurm_cloud.Slurm()
        feasible = cloud._get_feasible_launchable_resources(resources)

        assert len(feasible.resources_list) == 1
        resource = feasible.resources_list[0]

        instance_type = slurm_utils.SlurmInstanceType.from_instance_type(
            resource.instance_type)
        assert instance_type.cpus is None
        assert instance_type.memory is None
        assert instance_type.accelerator_count == gpu_count
        assert instance_type.accelerator_type == 'H200'

    @pytest.mark.parametrize(
        'accelerators,cpus,memory,expected_cpus,expected_memory',
        [
            # Various GPU types without explicit CPU/memory: GPU-only
            ({
                'H200': 2
            }, None, None, None, None),
            ({
                'A100': 2
            }, None, None, None, None),
            ({
                'H100': 2
            }, None, None, None, None),
            ({
                'A10G': 2
            }, None, None, None, None),
            # Explicit CPU override (memory scales from CPU)
            ({
                'H200': 2
            }, '16', None, 16, 64.0),
            # Explicit memory override (CPU uses default)
            ({
                'H200': 1
            }, None, '32', 4, 32.0),
            # Both CPU and memory override
            ({
                'H200': 2
            }, '32', '64', 32, 64.0),
            # Memory with '+' suffix
            ({
                'H200': 1
            }, None, '32+', 4, 32.0),
            # CPU-only instance (basic defaults)
            (None, None, None, 2, 2.0),
        ])
    @patch('sky.clouds.slurm.Slurm.regions_with_offering')
    def test_resource_allocation_scenarios(self, mock_regions, accelerators,
                                           cpus, memory, expected_cpus,
                                           expected_memory):
        """Test various resource allocation scenarios including GPU types and overrides."""
        mock_region = mock.MagicMock()
        mock_region.name = 'test-cluster'
        mock_regions.return_value = [mock_region]

        kwargs = {'cloud': slurm_cloud.Slurm()}
        if accelerators:
            kwargs['accelerators'] = accelerators
        if cpus:
            kwargs['cpus'] = cpus
        if memory:
            kwargs['memory'] = memory

        resources = resources_lib.Resources(**kwargs)
        cloud = slurm_cloud.Slurm()
        feasible = cloud._get_feasible_launchable_resources(resources)

        resource = feasible.resources_list[0]
        instance_type = slurm_utils.SlurmInstanceType.from_instance_type(
            resource.instance_type)

        assert instance_type.cpus == expected_cpus
        assert instance_type.memory == expected_memory


class TestGPUOnlyInstanceType:
    """Test that GPU-only instance types (no CPU/memory) are supported.

    When the user requests GPUs without specifying CPU/memory, the instance
    type should be GPU-only (e.g. 'GPU:1' instead of '4CPU--16GB--GPU:1').
    This allows the Slurm scheduler to auto-allocate CPU/memory per GPU.
    """

    def test_gpu_only_instance_type_name(self):
        """SlurmInstanceType with cpus=None, memory=None produces 'GPU:1'."""
        inst = slurm_utils.SlurmInstanceType(cpus=None,
                                             memory=None,
                                             accelerator_count=1,
                                             accelerator_type='GPU')
        assert inst.name == 'GPU:1'

    def test_gpu_only_instance_type_name_typed(self):
        """SlurmInstanceType with cpus=None, memory=None, typed GPU."""
        inst = slurm_utils.SlurmInstanceType(cpus=None,
                                             memory=None,
                                             accelerator_count=2,
                                             accelerator_type='H100')
        assert inst.name == 'H100:2'

    def test_gpu_only_is_valid(self):
        """GPU-only instance type names should be valid."""
        assert slurm_utils.SlurmInstanceType.is_valid_instance_type('GPU:1')
        assert slurm_utils.SlurmInstanceType.is_valid_instance_type('H100:4')

    def test_gpu_only_roundtrip(self):
        """GPU-only instance type should parse back to cpus=None, memory=None."""
        inst = slurm_utils.SlurmInstanceType.from_instance_type('GPU:1')
        assert inst.cpus is None
        assert inst.memory is None
        assert inst.accelerator_count == 1
        assert inst.accelerator_type == 'GPU'

    def test_cpu_mem_instance_type_unchanged(self):
        """Existing CPU/memory instance types still work."""
        inst = slurm_utils.SlurmInstanceType.from_instance_type(
            '4CPU--16GB--GPU:1')
        assert inst.cpus == 4.0
        assert inst.memory == 16.0
        assert inst.accelerator_count == 1
        assert inst.accelerator_type == 'GPU'

    def test_cpu_only_instance_type_unchanged(self):
        """CPU-only (no GPU) instance types still work."""
        inst = slurm_utils.SlurmInstanceType.from_instance_type('2CPU--8GB')
        assert inst.cpus == 2.0
        assert inst.memory == 8.0
        assert inst.accelerator_count is None

    def test_get_vcpus_mem_returns_none_for_gpu_only(self):
        """get_vcpus_mem_from_instance_type returns (None, None) for GPU-only."""
        vcpus, mem = slurm_cloud.Slurm.get_vcpus_mem_from_instance_type('GPU:1')
        assert vcpus is None
        assert mem is None

    def test_catalog_gpu_no_cpu_mem_gives_gpu_only_type(self):
        """Catalog generates GPU-only instance type when user omits CPU/mem."""
        from sky.catalog import slurm_catalog
        instance_type = slurm_catalog.get_default_instance_type(cpus=None,
                                                                memory=None)
        inst = slurm_utils.SlurmInstanceType.from_instance_type(instance_type)
        # CPU-only with no user spec should still have defaults
        assert inst.cpus is not None
        assert inst.memory is not None

    @patch('sky.clouds.slurm.Slurm.regions_with_offering')
    def test_feasible_resources_gpu_only(self, mock_regions):
        """When user requests only GPUs, feasible resources should have
        a GPU-only instance type with cpus=None, memory=None."""
        mock_region = mock.MagicMock()
        mock_region.name = 'test-cluster'
        mock_regions.return_value = [mock_region]

        resources = resources_lib.Resources(
            cloud=slurm_cloud.Slurm(),
            accelerators={'GPU': 1},
        )
        cloud = slurm_cloud.Slurm()
        feasible = cloud._get_feasible_launchable_resources(resources)

        assert len(feasible.resources_list) == 1
        resource = feasible.resources_list[0]
        inst = slurm_utils.SlurmInstanceType.from_instance_type(
            resource.instance_type)
        assert inst.cpus is None
        assert inst.memory is None
        assert inst.accelerator_count == 1

    @patch('sky.clouds.slurm.Slurm.regions_with_offering')
    def test_feasible_resources_gpu_with_explicit_cpu(self, mock_regions):
        """When user specifies CPU with GPU, instance type includes CPU/mem."""
        mock_region = mock.MagicMock()
        mock_region.name = 'test-cluster'
        mock_regions.return_value = [mock_region]

        resources = resources_lib.Resources(
            cloud=slurm_cloud.Slurm(),
            accelerators={'GPU': 1},
            cpus='16',
        )
        cloud = slurm_cloud.Slurm()
        feasible = cloud._get_feasible_launchable_resources(resources)

        resource = feasible.resources_list[0]
        inst = slurm_utils.SlurmInstanceType.from_instance_type(
            resource.instance_type)
        assert inst.cpus == 16.0
        assert inst.memory is not None
        assert inst.accelerator_count == 1


class TestGRESGPUParsing:
    """Test slurm_utils.get_gpu_type_and_count."""

    @pytest.mark.parametrize(
        'gres_str,expected_type,expected_count',
        [
            # Standard formats
            ('gpu:H100:8', 'H100', 8),
            ('gpu:a10g:4', 'a10g', 4),
            ('gpu:nvidia_h100_80gb_hbm3:8', 'nvidia_h100_80gb_hbm3', 8),
            ('gpu:A100_sxm4_40gb:4', 'A100_sxm4_40gb', 4),
            # No type
            ('gpu:8', None, 8),
            ('gpu:1', None, 1),
            # With extra Slurm info
            ('gpu:H100:8(S:0-1)', 'H100', 8),
            ('gpu:8(S:0)', None, 8),
            # Embedded in larger GRES string
            ('nic:1,gpu:H100:2,license:1', 'H100', 2),
            # Different casings for 'gpu'
            ('GPU:V100:1', 'V100', 1),
            ('Gpu:8', None, 8),
            # No match
            ('(null)', None, 0),
            ('N/A', None, 0),
            ('nic:1', None, 0),
            ('gpu', None, 0),
            ('gpu:', None, 0),
            ('not_a_gpu:8', None, 0),
            ('mygpu:8', None, 0),
        ])
    def test_get_gpu_type_and_count(self, gres_str, expected_type,
                                    expected_count):
        """Test that the helper correctly parses various GRES strings."""
        gpu_type, gpu_count = slurm_utils.get_gpu_type_and_count(gres_str)
        assert gpu_type == expected_type
        assert gpu_count == expected_count


SBATCH_TESTDATA_DIR = Path(__file__).parent / 'testdata' / 'slurm_sbatch'


def assert_sbatch_matches_snapshot(test_name: str,
                                   generated_script: str) -> None:
    """Compare generated sbatch script against snapshot file.

    Args:
        test_name: Name of the test (used to find snapshot file)
        generated_script: The script generated by _create_virtual_instance

    If UPDATE_SNAPSHOT=1 env var is set, updates the snapshot file instead
    of comparing.
    """
    snapshot_path = SBATCH_TESTDATA_DIR / f'{test_name}.sh'

    if os.environ.get('UPDATE_SNAPSHOT') == '1':
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(generated_script)
        print(f'Updated snapshot: {snapshot_path}')
        return

    if not snapshot_path.exists():
        pytest.fail(f'Snapshot file not found: {snapshot_path}\n'
                    f'Run with UPDATE_SNAPSHOT=1 to create it.')

    expected = snapshot_path.read_text()

    if generated_script != expected:
        import difflib
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            generated_script.splitlines(keepends=True),
            fromfile=f'{test_name}.sh (expected)',
            tofile=f'{test_name}.sh (actual)',
        )
        diff_text = ''.join(diff)
        pytest.fail(
            f'Generated script does not match snapshot: {snapshot_path}\n\n'
            f'Diff:\n{diff_text}\n\n'
            f'Run with UPDATE_SNAPSHOT=1 to update the snapshot.')


class TestSbatchOptionsPrecedence:
    """Test 4-level sbatch_options merge in make_deploy_resources_variables.

    Priority order (lowest to highest):
      1. ~/.sky/config.yaml  slurm.sbatch_options              (global)
      2. ~/.sky/config.yaml  slurm.cluster_configs.<c>.sbatch_options (cluster)
      3. ~/.sky/config.yaml  slurm.cluster_configs.<c>.partition_configs.<p>.sbatch_options (partition)
      4. Task YAML           config.slurm.sbatch_options        (task-level)

    Uses real config loading by writing to a tmp YAML file and reloading.
    """

    FAKE_SSH_CONFIG = {
        'hostname': '10.0.0.1',
        'port': '22',
        'user': 'slurm',
        'identityfile': ['/home/user/.ssh/id_rsa'],
    }

    def _load_config_and_get_sbatch_options(self,
                                            tmp_path,
                                            skypilot_config_dict,
                                            cluster_config_overrides=None):
        """Write config to tmp file, reload, and call make_deploy_resources_variables."""
        from sky import skypilot_config
        from sky.utils import yaml_utils

        # Write config dict to a tmp YAML file.
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(yaml_utils.dump_yaml_str(skypilot_config_dict))

        # Point config loading at our tmp file and reload.
        saved_global = skypilot_config._GLOBAL_CONFIG_PATH
        saved_project = skypilot_config._PROJECT_CONFIG_PATH
        saved_ctx = skypilot_config._global_config_context
        try:
            skypilot_config._GLOBAL_CONFIG_PATH = str(config_path)
            skypilot_config._PROJECT_CONFIG_PATH = str(tmp_path /
                                                       'nonexistent.yaml')
            skypilot_config._global_config_context = (
                skypilot_config.ConfigContext())
            skypilot_config.reload_config()

            cloud = slurm_cloud.Slurm()

            mock_resources = mock.MagicMock(unsafe=True)
            mock_resources.zone = 'gpu'
            mock_resources.instance_type = '4CPU--16GB'
            mock_resources.assert_launchable.return_value = mock_resources
            mock_resources.extract_docker_image.return_value = None
            mock_resources.cluster_config_overrides = (cluster_config_overrides
                                                       or {})

            region = mock.MagicMock()
            region.name = 'mycluster'
            zone_mock = mock.MagicMock()
            zone_mock.name = 'gpu'

            mock_ssh_config = mock.MagicMock()
            mock_ssh_config.lookup.return_value = self.FAKE_SSH_CONFIG

            with patch(
                    'sky.clouds.slurm.slurm_utils.get_slurm_ssh_config',
                    return_value=mock_ssh_config), \
                 patch(
                    'sky.clouds.slurm.slurm_utils.get_partitions',
                    return_value=['gpu', 'cpu']), \
                 patch(
                    'sky.clouds.slurm.slurm_utils.resolve_gres_gpu_type',
                    side_effect=lambda cluster, t, count=1, partition=None: t), \
                 patch(
                    'sky.clouds.slurm.slurm_utils.resolve_container_runtime',
                    return_value=None):
                deploy_vars = cloud.make_deploy_resources_variables(
                    resources=mock_resources,
                    cluster_name=mock.MagicMock(),
                    region=region,
                    zones=[zone_mock],
                    num_nodes=1,
                )
            return deploy_vars['sbatch_options']
        finally:
            skypilot_config._GLOBAL_CONFIG_PATH = saved_global
            skypilot_config._PROJECT_CONFIG_PATH = saved_project
            skypilot_config._global_config_context = saved_ctx

    def test_global_only(self, tmp_path):
        """Level 1: Only global sbatch_options set."""
        result = self._load_config_and_get_sbatch_options(
            tmp_path, {
                'slurm': {
                    'sbatch_options': {
                        'account': 'research',
                        'qos': 'normal',
                    },
                },
            })
        assert result == {'account': 'research', 'qos': 'normal'}

    def test_cluster_overrides_global(self, tmp_path):
        """Level 2 overrides level 1 for same key; new keys are merged."""
        result = self._load_config_and_get_sbatch_options(
            tmp_path, {
                'slurm': {
                    'sbatch_options': {
                        'account': 'global-account',
                        'qos': 'normal',
                    },
                    'cluster_configs': {
                        'mycluster': {
                            'sbatch_options': {
                                'account': 'cluster-account',
                                'constraint': 'skylake',
                            },
                        },
                    },
                },
            })
        assert result == {
            'account': 'cluster-account',
            'qos': 'normal',
            'constraint': 'skylake',
        }

    def test_partition_overrides_cluster_and_global(self, tmp_path):
        """Level 3 overrides levels 1 and 2 for same key."""
        result = self._load_config_and_get_sbatch_options(
            tmp_path, {
                'slurm': {
                    'sbatch_options': {
                        'account': 'global-account',
                        'qos': 'normal',
                    },
                    'cluster_configs': {
                        'mycluster': {
                            'sbatch_options': {
                                'account': 'cluster-account',
                            },
                            'partition_configs': {
                                'gpu': {
                                    'sbatch_options': {
                                        'account': 'partition-account',
                                        'exclusive': True,
                                    },
                                },
                            },
                        },
                    },
                },
            })
        assert result == {
            'account': 'partition-account',
            'qos': 'normal',
            'exclusive': True,
        }

    def test_task_overrides_all(self, tmp_path):
        """Level 4 (task YAML) overrides all config.yaml levels."""
        result = self._load_config_and_get_sbatch_options(
            tmp_path,
            skypilot_config_dict={
                'slurm': {
                    'sbatch_options': {
                        'account': 'global-account',
                        'qos': 'normal',
                    },
                    'cluster_configs': {
                        'mycluster': {
                            'sbatch_options': {
                                'account': 'cluster-account',
                            },
                            'partition_configs': {
                                'gpu': {
                                    'sbatch_options': {
                                        'account': 'partition-account',
                                        'exclusive': True,
                                    },
                                },
                            },
                        },
                    },
                },
            },
            cluster_config_overrides={
                'slurm': {
                    'sbatch_options': {
                        'account': 'task-account',
                        'nice': 100,
                    },
                },
            },
        )
        assert result == {
            'account': 'task-account',
            'qos': 'normal',
            'exclusive': True,
            'nice': 100,
        }

    def test_task_only(self, tmp_path):
        """Level 4 alone, no config.yaml sbatch_options at all."""
        result = self._load_config_and_get_sbatch_options(
            tmp_path,
            skypilot_config_dict={},
            cluster_config_overrides={
                'slurm': {
                    'sbatch_options': {
                        'account': 'task-only',
                    },
                },
            },
        )
        assert result == {'account': 'task-only'}

    def test_no_sbatch_options(self, tmp_path):
        """No sbatch_options at any level."""
        result = self._load_config_and_get_sbatch_options(
            tmp_path, skypilot_config_dict={})
        assert result == {}

    def test_task_cluster_specific_override(self, tmp_path):
        """Task YAML with cluster-specific sbatch_options."""
        result = self._load_config_and_get_sbatch_options(
            tmp_path,
            skypilot_config_dict={
                'slurm': {
                    'sbatch_options': {
                        'account': 'global-account',
                    },
                },
            },
            cluster_config_overrides={
                'slurm': {
                    'cluster_configs': {
                        'mycluster': {
                            'sbatch_options': {
                                'account': 'task-cluster-account',
                            },
                        },
                    },
                },
            },
        )
        assert result == {'account': 'task-cluster-account'}


class TestSlurmProvisionTimeout:
    """Test conditional provision timeout logic in make_deploy_resources_variables.

    When auto-selecting across partitions (no zone specified), timeout is 600s
    to enable failover. When user specifies a partition, timeout is -1
    (unlimited). Config overrides always take precedence.
    """

    FAKE_SSH_CONFIG = {
        'hostname': '10.0.0.1',
        'port': '22',
        'user': 'slurm',
        'identityfile': ['/home/user/.ssh/id_rsa'],
    }

    def _make_deploy_vars(self, zone, config_return):
        """Call make_deploy_resources_variables with mocked resources."""
        cloud = slurm_cloud.Slurm()

        mock_resources = mock.MagicMock(unsafe=True)
        mock_resources.zone = zone
        mock_resources.instance_type = '4CPU--16GB'
        mock_resources.assert_launchable.return_value = mock_resources
        mock_resources.extract_docker_image.return_value = None

        region = mock.MagicMock()
        region.name = 'test-cluster'
        zones_list = [mock.MagicMock(name=zone)] if zone else None

        mock_ssh_config = mock.MagicMock()
        mock_ssh_config.lookup.return_value = self.FAKE_SSH_CONFIG

        with patch('sky.clouds.slurm.skypilot_config.get_effective_region_config',
                   return_value=config_return) as mock_config, \
             patch('sky.clouds.slurm.slurm_utils.get_slurm_ssh_config',
                   return_value=mock_ssh_config), \
             patch('sky.clouds.slurm.slurm_utils.get_partitions',
                   return_value=['default', 'gpu', 'cpu']), \
             patch('sky.clouds.slurm.slurm_utils.resolve_gres_gpu_type',
                   side_effect=lambda cluster, t, count=1, partition=None: t), \
             patch('sky.clouds.slurm.slurm_utils.resolve_container_runtime',
                   return_value=None):
            deploy_vars = cloud.make_deploy_resources_variables(
                resources=mock_resources,
                cluster_name=mock.MagicMock(),
                region=region,
                zones=zones_list,
                num_nodes=1,
            )
        return deploy_vars, mock_config

    @pytest.mark.parametrize('zone,config_return,expected_timeout', [
        (None, None, 120),
        ('gpu', None, 86400),
        (None, 1800, 1800),
        ('gpu', 30, 30),
    ])
    def test_provision_timeout(self, zone, config_return, expected_timeout):
        """Test provision_timeout matches expected value."""
        deploy_vars, mock_config = self._make_deploy_vars(zone, config_return)
        assert deploy_vars['provision_timeout'] == expected_timeout
        mock_config.assert_any_call(cloud='slurm',
                                    region='test-cluster',
                                    keys=('provision_timeout',),
                                    default_value=None)
        # poll_interval is also read via get_effective_region_config
        mock_config.assert_any_call(cloud='slurm',
                                    region='test-cluster',
                                    keys=('poll_interval',),
                                    default_value=10)


class TestProvisionTimeoutPassthrough:
    """Test that provision_timeout from provider_config is used in
    _create_virtual_instance."""

    @pytest.fixture(autouse=True)
    def _mock_db_path(self, tmp_path):
        """Use a temp DB for SlurmJobInfo in all tests."""
        db_path = str(tmp_path / 'slurm_test.db')
        with patch('sky.adaptors.slurm.SlurmJobInfo._get_db_path',
                   return_value=db_path):
            yield

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_default_timeout_passthrough(self, mock_ssh_runner,
                                         mock_slurm_client,
                                         mock_get_partition_info,
                                         mock_get_proctrack_type,
                                         mock_wait_for_job_nodes):
        """When provider_config has provision_timeout (default 120s),
        _wait_for_job_nodes receives that value."""
        from sky.adaptors.slurm import SlurmPartition
        from sky.provision import common

        mock_get_partition_info.return_value = SlurmPartition(name='gpu',
                                                              is_default=False,
                                                              maxtime=7 * 24 *
                                                              60 * 60)
        mock_get_proctrack_type.return_value = 'cgroup'

        mock_client = mock.MagicMock()
        mock_client.query_jobs.return_value = []
        mock_client.get_job_nodes.return_value = (['node1'], {
            'node1': '10.0.0.5'
        })
        mock_client.get_partition_qos_max_wall.return_value = None
        mock_slurm_client.return_value = mock_client

        mock_runner = mock.MagicMock()
        mock_runner.run.return_value = (0, '', '')
        mock_runner.get_remote_home_dir.return_value = '/home/testuser'
        mock_ssh_runner.return_value = mock_runner

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'gpu',
                'provision_timeout': 120,
            },
            authentication_config={},
            docker_config={},
            node_config={
                'cpus': 2,
                'memory': 8,
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        slurm_instance._create_virtual_instance(
            region='us-west-2',
            cluster_name='test-timeout',
            cluster_name_on_cloud='test-timeout',
            config=config,
        )

        mock_wait_for_job_nodes.assert_called_once_with(
            mock.ANY,
            mock_client,
            mock.ANY,
            120,
            'gpu',
            mock.ANY,
            poll_interval=10,
            sbatch_log_dir=mock.ANY,
        )
        # get_job_nodes should be called without wait params
        mock_client.get_job_nodes.assert_called_once_with(mock.ANY)

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_explicit_timeout_passthrough(self, mock_ssh_runner,
                                          mock_slurm_client,
                                          mock_get_partition_info,
                                          mock_get_proctrack_type,
                                          mock_wait_for_job_nodes):
        """When provider_config has provision_timeout, it passes through."""
        from sky.adaptors.slurm import SlurmPartition
        from sky.provision import common

        mock_get_partition_info.return_value = SlurmPartition(name='gpu',
                                                              is_default=False,
                                                              maxtime=7 * 24 *
                                                              60 * 60)
        mock_get_proctrack_type.return_value = 'cgroup'

        mock_client = mock.MagicMock()
        mock_client.query_jobs.return_value = []
        mock_client.get_job_nodes.return_value = (['node1'], {
            'node1': '10.0.0.5'
        })
        mock_client.get_partition_qos_max_wall.return_value = None
        mock_slurm_client.return_value = mock_client

        mock_runner = mock.MagicMock()
        mock_runner.run.return_value = (0, '', '')
        mock_runner.get_remote_home_dir.return_value = '/home/testuser'
        mock_ssh_runner.return_value = mock_runner

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'gpu',
                'provision_timeout': 120,
            },
            authentication_config={},
            docker_config={},
            node_config={
                'cpus': 2,
                'memory': 8,
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        slurm_instance._create_virtual_instance(
            region='us-west-2',
            cluster_name='test-timeout-explicit',
            cluster_name_on_cloud='test-timeout-explicit',
            config=config,
        )

        # provision_timeout=120 should be passed to _wait_for_job_nodes
        mock_wait_for_job_nodes.assert_called_once_with(
            mock.ANY,
            mock_client,
            mock.ANY,
            120,
            'gpu',
            mock.ANY,
            poll_interval=10,
            sbatch_log_dir=mock.ANY,
        )
        # get_job_nodes should be called without wait params
        mock_client.get_job_nodes.assert_called_once_with(mock.ANY)

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_poll_interval_passthrough(self, mock_ssh_runner, mock_slurm_client,
                                       mock_get_partition_info,
                                       mock_get_proctrack_type,
                                       mock_wait_for_job_nodes):
        """When provider_config has poll_interval, it passes through."""
        from sky.adaptors.slurm import SlurmPartition
        from sky.provision import common

        mock_get_partition_info.return_value = SlurmPartition(name='gpu',
                                                              is_default=False,
                                                              maxtime=7 * 24 *
                                                              60 * 60)
        mock_get_proctrack_type.return_value = 'cgroup'

        mock_client = mock.MagicMock()
        mock_client.query_jobs.return_value = []
        mock_client.get_job_nodes.return_value = (['node1'], {
            'node1': '10.0.0.5'
        })
        mock_client.get_partition_qos_max_wall.return_value = None
        mock_slurm_client.return_value = mock_client

        mock_runner = mock.MagicMock()
        mock_runner.run.return_value = (0, '', '')
        mock_runner.get_remote_home_dir.return_value = '/home/testuser'
        mock_ssh_runner.return_value = mock_runner

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'gpu',
                'provision_timeout': 120,
                'poll_interval': 30,
            },
            authentication_config={},
            docker_config={},
            node_config={
                'cpus': 2,
                'memory': 8,
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        slurm_instance._create_virtual_instance(
            region='us-west-2',
            cluster_name='test-poll-interval',
            cluster_name_on_cloud='test-poll-interval',
            config=config,
        )

        mock_wait_for_job_nodes.assert_called_once_with(
            mock.ANY,
            mock_client,
            mock.ANY,
            120,
            'gpu',
            mock.ANY,
            poll_interval=30,
            sbatch_log_dir=mock.ANY,
        )


class TestCreateVirtualInstance:
    """Test slurm_instance._create_virtual_instance() script generation."""

    @pytest.fixture(autouse=True)
    def _mock_db_path(self, tmp_path):
        """Use a temp DB for SlurmJobInfo in all tests."""
        db_path = str(tmp_path / 'slurm_test.db')
        with patch('sky.adaptors.slurm.SlurmJobInfo._get_db_path',
                   return_value=db_path):
            yield

    def _setup_mocks(self, mock_ssh_runner, mock_slurm_client,
                     mock_get_partition_info, partition_name):
        """Configure standard mocks for _create_virtual_instance tests."""
        from sky.adaptors.slurm import SlurmPartition

        mock_get_partition_info.return_value = SlurmPartition(
            name=partition_name, is_default=False, maxtime=7 * 24 * 60 * 60)

        mock_client = mock.MagicMock()
        mock_client.query_jobs.return_value = []
        mock_client.get_job_nodes.return_value = (['node1'], {
            'node1': '10.0.0.5'
        })
        mock_client.get_partition_qos_max_wall.return_value = None
        mock_slurm_client.return_value = mock_client

        mock_runner = mock.MagicMock()
        mock_runner.run.return_value = (0, '', '')
        mock_runner.get_remote_home_dir.return_value = '/home/testuser'
        mock_ssh_runner.return_value = mock_runner

    def _run_and_capture_script(self, cluster_name, config):
        """Run _create_virtual_instance and capture the generated script."""
        written_script = None

        def capture_write(content):
            nonlocal written_script
            written_script = content

        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = mock.MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.write.side_effect = capture_write
            mock_tempfile.return_value = mock_file

            slurm_instance._create_virtual_instance(
                region='us-west-2',
                cluster_name=cluster_name,
                cluster_name_on_cloud=cluster_name,
                config=config,
            )

        assert written_script is not None, 'Script was not written'
        return written_script

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_container_script_format(self, mock_ssh_runner, mock_slurm_client,
                                     mock_get_partition_info,
                                     mock_get_proctrack_type,
                                     mock_wait_for_job_nodes):
        """Test that sbatch provision script for containers is correct."""
        from sky.provision import common

        self._setup_mocks(mock_ssh_runner, mock_slurm_client,
                          mock_get_partition_info, 'gpu')
        mock_get_proctrack_type.return_value = 'cgroup'

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'gpu',
                'provision_timeout': 300,
            },
            authentication_config={},
            docker_config={},
            node_config={
                'cpus': 4,
                'memory': 16,
                'accelerator_type': 'A100',
                'accelerator_count': 2,
                'image_id': 'nvcr.io/nvidia/pytorch:24.01-py3',
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        written_script = self._run_and_capture_script('test-cluster', config)
        assert_sbatch_matches_snapshot('containers', written_script)

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_non_container_script_format(self, mock_ssh_runner,
                                         mock_slurm_client,
                                         mock_get_partition_info,
                                         mock_get_proctrack_type,
                                         mock_wait_for_job_nodes):
        """Test that sbatch provision script without containers is correct."""
        from sky.provision import common

        self._setup_mocks(mock_ssh_runner, mock_slurm_client,
                          mock_get_partition_info, 'cpus')
        mock_get_proctrack_type.return_value = 'cgroup'

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'cpus',
                'provision_timeout': 300,
            },
            authentication_config={},
            docker_config={},
            node_config={
                'cpus': 2,
                'memory': 8,
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        written_script = self._run_and_capture_script(
            'test-cluster-no-container', config)
        assert_sbatch_matches_snapshot('basic', written_script)


class TestSbatchOmitCpuMemWhenUnspecified:
    """Test that --cpus-per-task and --mem are omitted from sbatch when
    the user did not explicitly request CPU/memory.

    On clusters like Isambard, each GPU automatically allocates a full
    superchip (72 cores + 115GB RAM). Passing explicit --cpus-per-task
    and --mem overrides this, giving a tiny allocation instead.

    Expected behavior:
    - GPUs requested, no explicit CPU/memory → omit --cpus-per-task/--mem
    - GPUs requested, explicit CPU/memory → include --cpus-per-task/--mem
    - No GPUs, explicit CPU/memory → include --cpus-per-task/--mem
    """

    @pytest.fixture(autouse=True)
    def _mock_db_path(self, tmp_path):
        """Use a temp DB for SlurmJobInfo in all tests."""
        db_path = str(tmp_path / 'slurm_test.db')
        with patch('sky.adaptors.slurm.SlurmJobInfo._get_db_path',
                   return_value=db_path):
            yield

    def _setup_mocks(self, mock_ssh_runner, mock_slurm_client,
                     mock_get_partition_info):
        """Configure standard mocks for _create_virtual_instance tests."""
        from sky.adaptors.slurm import SlurmPartition

        mock_get_partition_info.return_value = SlurmPartition(name='gpu',
                                                              is_default=False,
                                                              maxtime=7 * 24 *
                                                              60 * 60)

        mock_client = mock.MagicMock()
        mock_client.query_jobs.return_value = []
        mock_client.get_job_nodes.return_value = (['node1'], {
            'node1': '10.0.0.5'
        })
        mock_client.get_partition_qos_max_wall.return_value = None
        mock_slurm_client.return_value = mock_client

        mock_runner = mock.MagicMock()
        mock_runner.run.return_value = (0, '', '')
        mock_runner.get_remote_home_dir.return_value = '/home/testuser'
        mock_ssh_runner.return_value = mock_runner

    def _run_and_capture_script(self, cluster_name, config):
        """Run _create_virtual_instance and capture the generated script."""
        written_script = None

        def capture_write(content):
            nonlocal written_script
            written_script = content

        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = mock.MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.write.side_effect = capture_write
            mock_tempfile.return_value = mock_file

            slurm_instance._create_virtual_instance(
                region='us-west-2',
                cluster_name=cluster_name,
                cluster_name_on_cloud=cluster_name,
                config=config,
            )

        assert written_script is not None, 'Script was not written'
        return written_script

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_gpu_without_explicit_cpu_mem_omits_sbatch_flags(
            self, mock_ssh_runner, mock_slurm_client, mock_get_partition_info,
            mock_get_proctrack_type, mock_wait_for_job_nodes):
        """When GPUs are requested without explicit CPU/memory, the sbatch
        script should NOT contain --cpus-per-task or --mem directives.
        This lets the scheduler auto-allocate resources per GPU."""
        from sky.provision import common

        self._setup_mocks(mock_ssh_runner, mock_slurm_client,
                          mock_get_partition_info)
        mock_get_proctrack_type.return_value = 'cgroup'

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'gpu',
                'provision_timeout': 300,
            },
            authentication_config={},
            docker_config={},
            node_config={
                # cpus and memory are None = user didn't specify them
                'cpus': None,
                'memory': None,
                'accelerator_type': 'GPU',
                'accelerator_count': 1,
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        script = self._run_and_capture_script('test-gpu-no-cpu-mem', config)
        assert '--cpus-per-task' not in script, (
            'sbatch script should not contain --cpus-per-task when user '
            'did not specify CPUs (let scheduler auto-allocate per GPU)')
        assert '#SBATCH --mem=' not in script, (
            'sbatch script should not contain --mem when user '
            'did not specify memory (let scheduler auto-allocate per GPU)')
        # GPU directive should still be present
        assert '--gres=gpu:1' in script

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_gpu_with_explicit_cpu_mem_includes_sbatch_flags(
            self, mock_ssh_runner, mock_slurm_client, mock_get_partition_info,
            mock_get_proctrack_type, mock_wait_for_job_nodes):
        """When GPUs are requested WITH explicit CPU/memory, the sbatch
        script SHOULD contain --cpus-per-task and --mem directives."""
        from sky.provision import common

        self._setup_mocks(mock_ssh_runner, mock_slurm_client,
                          mock_get_partition_info)
        mock_get_proctrack_type.return_value = 'cgroup'

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'gpu',
                'provision_timeout': 300,
            },
            authentication_config={},
            docker_config={},
            node_config={
                'cpus': 16,
                'memory': 64,
                'accelerator_type': 'GPU',
                'accelerator_count': 2,
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        script = self._run_and_capture_script('test-gpu-with-cpu-mem', config)
        assert '#SBATCH --cpus-per-task=16' in script, (
            'sbatch script should contain --cpus-per-task when user '
            'explicitly specified CPUs')
        assert '#SBATCH --mem=64G' in script, (
            'sbatch script should contain --mem when user '
            'explicitly specified memory')
        assert '--gres=gpu:2' in script

    @patch('sky.provision.slurm.instance._wait_for_job_nodes')
    @patch('sky.provision.slurm.instance.slurm_utils.get_proctrack_type')
    @patch('sky.provision.slurm.instance.slurm_utils.get_partition_info')
    @patch('sky.provision.slurm.instance.slurm.SlurmClient')
    @patch('sky.provision.slurm.instance.command_runner.SSHCommandRunner')
    def test_no_gpu_with_explicit_cpu_mem_includes_sbatch_flags(
            self, mock_ssh_runner, mock_slurm_client, mock_get_partition_info,
            mock_get_proctrack_type, mock_wait_for_job_nodes):
        """CPU-only jobs with explicit CPU/memory should include the flags."""
        from sky.provision import common

        self._setup_mocks(mock_ssh_runner, mock_slurm_client,
                          mock_get_partition_info)
        mock_get_proctrack_type.return_value = 'cgroup'

        config = common.ProvisionConfig(
            provider_config={
                'ssh': {
                    'hostname': 'login.example.com',
                    'port': '22',
                    'user': 'testuser',
                    'private_key': '/path/to/key',
                },
                'cluster': 'test-slurm',
                'partition': 'cpu',
                'provision_timeout': 300,
            },
            authentication_config={},
            docker_config={},
            node_config={
                'cpus': 8,
                'memory': 32,
            },
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        )

        script = self._run_and_capture_script('test-cpu-only', config)
        assert '#SBATCH --cpus-per-task=8' in script
        assert '#SBATCH --mem=32G' in script


class TestDeployVarsCpuMemPassthrough:
    """Test that make_deploy_resources_variables passes None for CPU/memory
    when the user did not specify them, so the template can conditionally
    omit --cpus-per-task and --mem from the sbatch script."""

    FAKE_SSH_CONFIG = {
        'hostname': '10.0.0.1',
        'port': '22',
        'user': 'slurm',
        'identityfile': ['/home/user/.ssh/id_rsa'],
    }

    def _get_deploy_vars(self, instance_type, user_cpus, user_memory):
        """Call make_deploy_resources_variables with controlled resources."""
        cloud = slurm_cloud.Slurm()

        mock_resources = mock.MagicMock(unsafe=True)
        mock_resources.zone = 'gpu'
        mock_resources.instance_type = instance_type
        mock_resources.assert_launchable.return_value = mock_resources
        mock_resources.extract_docker_image.return_value = None
        mock_resources.cluster_config_overrides = {}
        # These are None when not user-specified
        mock_resources.cpus = user_cpus
        mock_resources.memory = user_memory

        region = mock.MagicMock()
        region.name = 'mycluster'
        zone_mock = mock.MagicMock()
        zone_mock.name = 'gpu'

        mock_ssh_config = mock.MagicMock()
        mock_ssh_config.lookup.return_value = self.FAKE_SSH_CONFIG

        with patch('sky.clouds.slurm.slurm_utils.get_slurm_ssh_config',
                   return_value=mock_ssh_config), \
             patch('sky.clouds.slurm.slurm_utils.get_partitions',
                   return_value=['gpu', 'cpu']), \
             patch('sky.clouds.slurm.slurm_utils.resolve_gres_gpu_type',
                   side_effect=lambda cluster, t, count=1, partition=None: t), \
             patch('sky.clouds.slurm.slurm_utils.resolve_container_runtime',
                   return_value=None):
            return cloud.make_deploy_resources_variables(
                resources=mock_resources,
                cluster_name=mock.MagicMock(),
                region=region,
                zones=[zone_mock],
                num_nodes=1,
            )

    def test_gpu_no_explicit_cpu_mem_passes_none(self):
        """When user requests GPUs without specifying CPU/memory,
        deploy vars should have cpus=None and memory=None."""
        # GPU-only instance type: no CPU/memory in the name
        deploy_vars = self._get_deploy_vars(
            instance_type='GPU:1',
            user_cpus=None,
            user_memory=None,
        )
        assert deploy_vars['cpus'] is None, (
            'cpus should be None when user did not specify')
        assert deploy_vars['memory'] is None, (
            'memory should be None when user did not specify')

    def test_gpu_with_explicit_cpu_mem_passes_values(self):
        """When user explicitly requests CPU/memory, deploy vars should
        have those values."""
        deploy_vars = self._get_deploy_vars(
            instance_type='16CPU--64GB--GPU:2',
            user_cpus='16',
            user_memory='64',
        )
        assert deploy_vars['cpus'] == '16.0'
        assert deploy_vars['memory'] == '64.0'

    def test_cpu_only_explicit_passes_values(self):
        """CPU-only instance with explicit resources should pass values."""
        deploy_vars = self._get_deploy_vars(
            instance_type='8CPU--32GB',
            user_cpus='8',
            user_memory='32',
        )
        assert deploy_vars['cpus'] == '8.0'
        assert deploy_vars['memory'] == '32.0'


class TestGetPendingJobCount:
    """Test SlurmClient.get_pending_job_count()."""

    @patch('sky.adaptors.slurm.SlurmClient._run_slurm_cmd')
    def test_basic_count(self, mock_run):
        """Test counting pending jobs in a partition."""
        mock_run.return_value = (0, '100\n101\n102\n', '')
        client = slurm.SlurmClient(is_inside_slurm_cluster=True)
        count = client.get_pending_job_count('gpu')
        assert count == 3

    @patch('sky.adaptors.slurm.SlurmClient._run_slurm_cmd')
    def test_exclude_own_job(self, mock_run):
        """Test excluding our own job from the count."""
        mock_run.return_value = (0, '100\n101\n102\n', '')
        client = slurm.SlurmClient(is_inside_slurm_cluster=True)
        count = client.get_pending_job_count('gpu', exclude_job_id='101')
        assert count == 2

    @patch('sky.adaptors.slurm.SlurmClient._run_slurm_cmd')
    def test_empty_queue(self, mock_run):
        """Test empty pending queue returns 0."""
        mock_run.return_value = (0, '', '')
        client = slurm.SlurmClient(is_inside_slurm_cluster=True)
        count = client.get_pending_job_count('gpu')
        assert count == 0

    @patch('sky.adaptors.slurm.SlurmClient._run_slurm_cmd')
    def test_command_failure(self, mock_run):
        """Test that command failure returns -1."""
        mock_run.return_value = (1, '', 'error')
        client = slurm.SlurmClient(is_inside_slurm_cluster=True)
        count = client.get_pending_job_count('gpu')
        assert count == -1

    @patch('sky.adaptors.slurm.SlurmClient._run_slurm_cmd')
    def test_only_own_job(self, mock_run):
        """Test that excluding our own job from single-entry queue returns 0."""
        mock_run.return_value = (0, '100\n', '')
        client = slurm.SlurmClient(is_inside_slurm_cluster=True)
        count = client.get_pending_job_count('gpu', exclude_job_id='100')
        assert count == 0


class TestOnPendingMessage:
    """Test the _on_pending callback message formatting."""

    @pytest.mark.parametrize('reason,pending_count,expected', [
        ('Resources', 12, 'Launching (pending: Resources, 12 others pending)'),
        ('Priority', 1, 'Launching (pending: Priority, 1 other pending)'),
        ('Resources', 0, 'Launching (pending: Resources)'),
        (None, 5, 'Launching (5 others pending)'),
        (None, None, 'Launching'),
        (None, 0, 'Launching'),
        ('Priority', None, 'Launching (pending: Priority)'),
    ])
    def test_on_pending_message_format(self, reason, pending_count, expected):
        """Test _on_pending callback formats messages correctly."""
        parts = []
        if reason:
            parts.append(f'pending: {reason}')
        if pending_count is not None and pending_count > 0:
            word = 'other' if pending_count == 1 else 'others'
            parts.append(f'{pending_count} {word} pending')
        if parts:
            msg = f'Launching ({", ".join(parts)})'
        else:
            msg = 'Launching'

        assert msg == expected


class TestHelperPathFunctions:
    """Test path helper functions in sky/provision/slurm/instance.py."""

    @pytest.mark.parametrize('base_dir,job_id,expected', [
        ('/home/user', '123', '/home/user/.sky_provision/slurm-123.out'),
        ('/fsx/ubuntu', '456', '/fsx/ubuntu/.sky_provision/slurm-456.out'),
        ('/home/user', '%j', '/home/user/.sky_provision/slurm-%j.out'),
    ])
    def test_sbatch_log_path(self, base_dir, job_id, expected):
        assert slurm_instance._sbatch_log_path(base_dir, job_id) == expected

    @pytest.mark.parametrize('base_dir,cluster,expected', [
        ('/home/user', 'my-cluster', '/home/user/.sky_clusters/my-cluster'),
        ('/fsx/ubuntu', 'test-abc123', '/fsx/ubuntu/.sky_clusters/test-abc123'),
    ])
    def test_sky_cluster_home_dir(self, base_dir, cluster, expected):
        assert slurm_instance._sky_cluster_home_dir(base_dir,
                                                    cluster) == expected

    @pytest.mark.parametrize('base_dir,cluster,expected', [
        ('/home/user', 'my-cluster', '/home/user/.sky_provision/my-cluster.sh'),
        ('~', 'my-cluster', '~/.sky_provision/my-cluster.sh'),
    ])
    def test_sbatch_provision_script_path(self, base_dir, cluster, expected):
        assert slurm_instance._sbatch_provision_script_path(base_dir,
                                                            cluster) == expected

    @pytest.mark.parametrize('tmpdir,cluster,expected', [
        (None, 'my-cluster', '/tmp/my-cluster'),
        ('/scratch/tmp', 'my-cluster', '/scratch/tmp/my-cluster'),
    ])
    def test_skypilot_runtime_dir(self, tmpdir, cluster, expected):
        assert slurm_instance._skypilot_runtime_dir(tmpdir, cluster) == expected


class TestGetEnv:
    """Test SlurmClient.get_env() parsing."""

    def test_parses_env_output(self):
        client = mock.MagicMock(spec=slurm.SlurmClient)
        client._run_slurm_cmd.return_value = (
            0, 'HOME=/home/ubuntu\nUSER=ubuntu\n', '')
        # Call the real method with the mocked client
        env = slurm.SlurmClient.get_env(client)
        assert env == {'HOME': '/home/ubuntu', 'USER': 'ubuntu'}

    def test_handles_values_with_equals(self):
        client = mock.MagicMock(spec=slurm.SlurmClient)
        client._run_slurm_cmd.return_value = (0,
                                              'PATH=/usr/bin:/bin\nFOO=a=b\n',
                                              '')
        env = slurm.SlurmClient.get_env(client)
        assert env['PATH'] == '/usr/bin:/bin'
        assert env['FOO'] == 'a=b'

    def test_command_failure_returns_empty(self):
        client = mock.MagicMock(spec=slurm.SlurmClient)
        client._run_slurm_cmd.return_value = (1, '', 'Connection refused')
        env = slurm.SlurmClient.get_env(client)
        assert env == {}


class TestExpandPathVars:
    """Test expand_path_vars with Python-side variable expansion."""

    REMOTE_ENV = {'USER': 'ubuntu', 'HOME': '/home/ubuntu'}

    def test_expands_dollar_var(self):
        result = slurm_utils.expand_path_vars('/fsx/$USER', self.REMOTE_ENV)
        assert result == '/fsx/ubuntu'

    def test_expands_braced_var(self):
        result = slurm_utils.expand_path_vars('/fsx/${USER}', self.REMOTE_ENV)
        assert result == '/fsx/ubuntu'

    def test_expands_multiple_vars(self):
        result = slurm_utils.expand_path_vars('$HOME/$USER', self.REMOTE_ENV)
        assert result == '/home/ubuntu/ubuntu'

    def test_unknown_var_left_unchanged(self):
        result = slurm_utils.expand_path_vars('/fsx/$UNKNOWN', self.REMOTE_ENV)
        assert result == '/fsx/$UNKNOWN'

    def test_no_vars_passthrough(self):
        result = slurm_utils.expand_path_vars('/fsx/ubuntu', self.REMOTE_ENV)
        assert result == '/fsx/ubuntu'

    def test_injection_not_expanded(self):
        # Even if someone puts shell metacharacters in config,
        # they are never sent to a shell — just literal string replacement.
        result = slurm_utils.expand_path_vars('/home/; rm -rf /',
                                              self.REMOTE_ENV)
        assert result == '/home/; rm -rf /'
