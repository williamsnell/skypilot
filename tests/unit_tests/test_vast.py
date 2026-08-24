"""Vast provisioning: docker_login_config must reach the ray template.

Vast is docker-native (the instance IS the container), so a private-registry
login travels through the provider config rather than a top-level ``docker:``
field. ``sky/clouds/vast.py`` must bind ``docker_login_config`` in
``make_deploy_resources_variables`` (mirroring ``yotta.py``), and
``sky/templates/vast-ray.yml.j2`` renders it under ``provider:``.

The end-to-end test composes the template variables the way
``backend_utils.write_cluster_config`` does -- the make_deploy output plus the
common backend block -- then renders the real template via ``fill_template``.
``write_cluster_config`` itself is not driven: it needs the catalog, credential
checks and (in the client-server model) the API server. So these tests cover the
make_deploy binding and the template render, not that surrounding machinery.

The common variable block is *derived from the template* (jinja2.meta), not
hand-listed, so it cannot drift as upstream adds template variables: the claim
under test becomes "of every name vast-ray.yml.j2 needs, docker_login_config is
the one make_deploy must supply."
"""
import os
from unittest import mock

import jinja2
from jinja2 import meta
import yaml

from sky.clouds import vast
from sky.provision import docker_utils
from sky.utils import common_utils
from sky.utils import resources_utils

_TEMPLATE_NAME = 'vast-ray.yml.j2'
# Resolve the template the same way common_utils.fill_template does, so the test
# tracks the real file regardless of cwd.
_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(common_utils.__file__)), 'templates',
    _TEMPLATE_NAME)

# The handful of template names the fixture must type on its own (not plain
# strings); everything else defaults to 'dummy'. docker_login_config is
# deliberately NOT here: it is always excluded or overridden by the caller, so
# typing it here would only mask a caller that forgot to.
_TYPE_OVERRIDES = {
    'num_nodes': 1,
    'disk_size': 40,
    'use_spot': False,
    'credentials': {},
    'initial_setup_commands': [],
    'create_instance_kwargs': {},
}


def _template_vars(exclude=()):
    """Every name vast-ray.yml.j2 needs, filled with a type-appropriate dummy.

    Derived from the template so it cannot drift as upstream adds variables.
    Names in ``exclude`` are omitted, so the caller controls which are unbound.
    """
    with open(_TEMPLATE_PATH, encoding='utf-8') as f:
        source = f.read()
    names = meta.find_undeclared_variables(jinja2.Environment().parse(source))
    return {
        name: _TYPE_OVERRIDES.get(name, 'dummy')
        for name in names
        if name not in exclude
    }


def _full_vars(docker_login_config):
    return {**_template_vars(), 'docker_login_config': docker_login_config}


def _deploy_vars(docker_login_config):
    """The real make_deploy_resources_variables output. resources/region are
    mocked so no catalog is needed; the value under test is whether the returned
    dict carries docker_login_config."""
    resources = mock.MagicMock(unsafe=True)
    resources.assert_launchable.return_value = resources
    resources.instance_type = '1x-RTX_4090-16'
    resources.image_id = None
    resources.cluster_config_overrides = {}
    resources.docker_login_config = docker_login_config
    region = mock.MagicMock()
    region.name = 'someregion'
    with mock.patch.object(vast.Vast,
                           'get_accelerators_from_instance_type',
                           return_value={}), \
         mock.patch('sky.clouds.vast.resources_utils.'
                    'make_ray_custom_resources_str', return_value=None), \
         mock.patch('sky.clouds.vast.skypilot_config.'
                    'get_effective_region_config',
                    side_effect=lambda **kw: kw.get('default_value')):
        return vast.Vast().make_deploy_resources_variables(
            resources=resources,
            cluster_name=resources_utils.ClusterName(display_name='vastcheck',
                                                     name_on_cloud='vastcheck'),
            region=region,
            zones=None,
            num_nodes=1)


def _render(tmp_path, variables):
    out = tmp_path / 'vast-ray.yml'
    common_utils.fill_template(_TEMPLATE_NAME, variables, str(out))
    return yaml.safe_load(out.read_text(encoding='utf-8'))


def test_template_vars_covers_template(tmp_path):
    """The derived fixture renders the real template with the credential block
    POPULATED -- so every name the template needs is exercised, not just the
    block-skipped path. If this fails, the deriving helper is broken, not the
    code under test."""
    login = docker_utils.DockerLoginConfig(username='dummy',
                                           password='dummy',
                                           server='index.docker.io')
    parsed = _render(tmp_path, _full_vars(login))
    assert isinstance(parsed, dict) and 'provider' in parsed


def test_vast_end_to_end_config_generation(tmp_path):
    """Credentials supplied via make_deploy reach the provider config. This is
    the test the file exists for: on williamsnell/fix-vast-docker-login the
    variables lack docker_login_config, so the render raises UndefinedError. The
    exclude makes make_deploy the only possible source of the name."""
    login = docker_utils.DockerLoginConfig(username='dummy',
                                           password='dummy',
                                           server='index.docker.io')
    variables = {
        **_template_vars(exclude=('docker_login_config',)),
        **_deploy_vars(login),
    }
    parsed = _render(tmp_path, variables)
    # Docker-native: the login sits under provider, NOT as a top-level `docker:`
    # field. A top-level docker key would trigger DockerInitializer host-side
    # setup (sky/provision/provisioner.py) -- the failure mode this design
    # avoids. This assertion pins the #9632 design dispute.
    assert 'docker' not in parsed
    dlc = parsed['provider']['docker_login_config']
    assert dlc['username'].strip() == 'dummy'
    assert dlc['password'].strip() == 'dummy'
    assert dlc['server'].strip() == 'index.docker.io'


def test_vast_template_renders_private_login(tmp_path):
    """A populated login renders under provider, and a multi-line password stays
    valid YAML -- the template's `| indent(6)` exists for exactly this and is
    easy to drop in a refactor."""
    login = docker_utils.DockerLoginConfig(username='dummy',
                                           password='line1\nline2',
                                           server='index.docker.io')
    parsed = _render(tmp_path, _full_vars(login))
    dlc = parsed['provider']['docker_login_config']
    assert dlc['username'].strip() == 'dummy'
    assert dlc['server'].strip() == 'index.docker.io'
    assert dlc['password'].strip() == 'line1\nline2'


def test_vast_template_public_image_skips_block(tmp_path):
    """Public image (docker_login_config=None): the block is skipped and the
    config carries no docker credentials anywhere (no-regression guard)."""
    parsed = _render(tmp_path, _full_vars(None))
    assert 'docker_login_config' not in parsed.get('provider', {})
    assert 'docker' not in parsed


def test_vast_make_deploy_binds_docker_login_config():
    """make_deploy_resources_variables copies resources.docker_login_config into
    the deploy variables (the one changed line); None passes through. A local
    regression guard on clouds/vast.py -- not, by itself, PR evidence."""
    login = docker_utils.DockerLoginConfig(username='dummy',
                                           password='dummy',
                                           server='index.docker.io')
    assert _deploy_vars(login)['docker_login_config'] is login
    assert _deploy_vars(None)['docker_login_config'] is None
