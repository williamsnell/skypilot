"""Unit tests for sky setup-slurm-ssh persistence and overlay merging."""
import os
import stat
import tempfile
import time

import pytest

from sky import exceptions
from sky.provision.slurm import utils as slurm_utils


@pytest.fixture
def slurm_dir(tmp_path, monkeypatch):
    """Set up a temporary Slurm config directory structure."""
    # Base config (shared / server-side)
    base_config = tmp_path / 'config'
    base_config.write_text('Host mycluster\n'
                           '    HostName login.example.com\n'
                           '    Port 22\n'
                           '    ContainerRuntime podman-hpc\n'
                           '\n'
                           'Host *\n'
                           '    ServerAliveInterval 60\n')

    users_dir = tmp_path / 'users'
    users_dir.mkdir()

    # Patch the module-level constants
    monkeypatch.setattr(slurm_utils, 'DEFAULT_SLURM_PATH', str(base_config))
    monkeypatch.setattr(slurm_utils, 'SLURM_USERS_DIR', str(users_dir))

    return tmp_path


class TestPersistSlurmSshCredentials:
    """Tests for persist_slurm_ssh_credentials()."""

    def test_writes_key_with_correct_permissions(self, slurm_dir):
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='PRIVATE_KEY_DATA',
            ssh_user='testuser',
        )
        key_path = os.path.join(user_dir, 'skypilot_slurm')
        assert os.path.isfile(key_path)
        assert open(key_path).read() == 'PRIVATE_KEY_DATA'
        assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600

    def test_directory_permissions(self, slurm_dir):
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='KEY',
            ssh_user='testuser',
        )
        assert stat.S_IMODE(os.stat(user_dir).st_mode) == 0o700

    def test_writes_certificate(self, slurm_dir):
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='KEY',
            ssh_user='testuser',
            certificate_content='CERT_DATA',
        )
        cert_path = os.path.join(user_dir, 'skypilot_slurm-cert.pub')
        assert os.path.isfile(cert_path)
        assert open(cert_path).read() == 'CERT_DATA'

    def test_generates_ssh_config_overlay(self, slurm_dir):
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='KEY',
            ssh_user='testuser',
            proxy_jump='user@jump.example:22',
        )
        config_path = os.path.join(user_dir, 'config')
        config_text = open(config_path).read()
        assert 'User testuser' in config_text
        assert 'IdentityFile' in config_text
        assert 'ProxyJump user@jump.example:22' in config_text
        assert 'IdentitiesOnly yes' in config_text

    def test_idempotent_overwrite(self, slurm_dir):
        """Re-running with different content overwrites cleanly."""
        slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='OLD_KEY',
            ssh_user='olduser',
        )
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='NEW_KEY',
            ssh_user='newuser',
        )
        key_path = os.path.join(user_dir, 'skypilot_slurm')
        assert open(key_path).read() == 'NEW_KEY'
        config_text = open(os.path.join(user_dir, 'config')).read()
        assert 'User newuser' in config_text

    def test_user_isolation(self, slurm_dir):
        """Two user_hashes get separate directories."""
        dir_a = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user-a',
            private_key_content='KEY_A',
            ssh_user='alice',
        )
        dir_b = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user-b',
            private_key_content='KEY_B',
            ssh_user='bob',
        )
        assert dir_a != dir_b
        assert open(os.path.join(dir_a, 'skypilot_slurm')).read() == 'KEY_A'
        assert open(os.path.join(dir_b, 'skypilot_slurm')).read() == 'KEY_B'

    def test_empty_user_hash_raises(self, slurm_dir):
        with pytest.raises(ValueError, match='user_hash is required'):
            slurm_utils.persist_slurm_ssh_credentials(
                user_hash='',
                private_key_content='KEY',
                ssh_user='testuser',
            )

    def test_cert_expiry_written(self, slurm_dir):
        expires = time.time() + 3600
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='KEY',
            ssh_user='testuser',
            cert_expires_at=expires,
        )
        expires_path = os.path.join(user_dir, 'cert_expires_at')
        assert os.path.isfile(expires_path)
        assert float(open(expires_path).read().strip()) == expires

    def test_cert_expiry_removed_when_none(self, slurm_dir):
        """If cert_expires_at is None and file exists, it's removed."""
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='KEY',
            ssh_user='testuser',
            cert_expires_at=time.time() + 3600,
        )
        expires_path = os.path.join(user_dir, 'cert_expires_at')
        assert os.path.isfile(expires_path)

        # Re-run without cert_expires_at
        slurm_utils.persist_slurm_ssh_credentials(
            user_hash='abc123',
            private_key_content='KEY',
            ssh_user='testuser',
            cert_expires_at=None,
        )
        assert not os.path.isfile(expires_path)


class TestGetSlurmSshConfigOverlay:
    """Tests for get_slurm_ssh_config() with per-user overlay merging."""

    def test_base_config_only(self, slurm_dir):
        """Without overlay, returns base config values."""
        config = slurm_utils.get_slurm_ssh_config(user_hash='nonexistent')
        result = config.lookup('mycluster')
        assert result['hostname'] == 'login.example.com'
        # No User in base config
        assert result.get('user') is None or result['user'] == os.getenv(
            'USER', '')

    def test_overlay_user_wins(self, slurm_dir):
        """Overlay User takes precedence over base config."""
        slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user1',
            private_key_content='KEY',
            ssh_user='alice',
        )
        config = slurm_utils.get_slurm_ssh_config(user_hash='user1')
        result = config.lookup('mycluster')
        assert result['user'] == 'alice'
        # Base config values preserved
        assert result['hostname'] == 'login.example.com'

    def test_overlay_identity_file(self, slurm_dir):
        """Overlay IdentityFile points to persisted key."""
        user_dir = slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user1',
            private_key_content='KEY',
            ssh_user='alice',
        )
        config = slurm_utils.get_slurm_ssh_config(user_hash='user1')
        result = config.lookup('mycluster')
        identity_files = result.get('identityfile', [])
        expected_key = os.path.join(user_dir, 'skypilot_slurm')
        assert expected_key in identity_files

    def test_overlay_proxy_jump(self, slurm_dir):
        """Overlay ProxyJump is included when set."""
        slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user1',
            private_key_content='KEY',
            ssh_user='alice',
            proxy_jump='alice@jump.example',
        )
        config = slurm_utils.get_slurm_ssh_config(user_hash='user1')
        result = config.lookup('mycluster')
        assert result.get('proxyjump') == 'alice@jump.example'

    def test_no_overlay_no_proxyjump(self, slurm_dir):
        """Without overlay, no proxyjump."""
        config = slurm_utils.get_slurm_ssh_config(user_hash='nonexistent')
        result = config.lookup('mycluster')
        assert result.get('proxyjump') is None


class TestCertExpiry:
    """Tests for certificate expiry checking."""

    def test_no_expiry_file(self, slurm_dir):
        """No expiry file returns None."""
        assert slurm_utils.check_cert_expiry('nonexistent') is None

    def test_valid_cert(self, slurm_dir):
        """Valid (future) expiry returns the timestamp."""
        future = time.time() + 3600
        slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user1',
            private_key_content='KEY',
            ssh_user='alice',
            cert_expires_at=future,
        )
        result = slurm_utils.check_cert_expiry('user1')
        assert result == future

    def test_expired_cert_raises(self, slurm_dir):
        """Expired cert raises SlurmSshSetupRequired."""
        past = time.time() - 3600
        slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user1',
            private_key_content='KEY',
            ssh_user='alice',
            cert_expires_at=past,
        )
        with pytest.raises(exceptions.SlurmSshSetupRequired, match='expired'):
            slurm_utils.check_cert_expiry('user1')

    def test_empty_user_hash(self, slurm_dir):
        assert slurm_utils.check_cert_expiry('') is None


class TestHasPersistedCredentials:
    """Tests for has_persisted_credentials()."""

    def test_no_credentials(self, slurm_dir):
        assert not slurm_utils.has_persisted_credentials('nonexistent')

    def test_with_credentials(self, slurm_dir):
        slurm_utils.persist_slurm_ssh_credentials(
            user_hash='user1',
            private_key_content='KEY',
            ssh_user='alice',
        )
        assert slurm_utils.has_persisted_credentials('user1')

    def test_empty_user_hash(self, slurm_dir):
        assert not slurm_utils.has_persisted_credentials('')


class TestMakeSlurmClientSetupRequired:
    """Tests that make_slurm_client_from_config raises the right error."""

    def test_raises_when_no_user(self, slurm_dir):
        """Missing User raises SlurmSshSetupRequired."""
        config = slurm_utils.get_slurm_ssh_config(user_hash='nonexistent')
        ssh_dict = config.lookup('mycluster')
        # Ensure no user is set
        ssh_dict.pop('user', None)
        with pytest.raises(exceptions.SlurmSshSetupRequired,
                           match='setup-slurm-ssh'):
            slurm_utils.make_slurm_client_from_config(ssh_dict, 'mycluster')


class TestParseCertExpiry:
    """Tests for _parse_cert_expiry in sdk.py."""

    def test_parses_standard_format(self):
        from sky.client.sdk import _parse_cert_expiry
        output = (
            '        Type: ssh-ed25519-cert-v01@openssh.com user certificate\n'
            '        Valid: from 2026-03-29T10:00:00 to 2026-03-30T14:00:00\n'
            '        Principals:\n')
        result = _parse_cert_expiry(output)
        assert result is not None
        # Should be March 30, 2026 14:00:00 as a timestamp
        assert result > 0

    def test_returns_none_for_no_valid_line(self):
        from sky.client.sdk import _parse_cert_expiry
        output = 'some random output\nno valid line here\n'
        assert _parse_cert_expiry(output) is None

    def test_returns_none_for_empty(self):
        from sky.client.sdk import _parse_cert_expiry
        assert _parse_cert_expiry('') is None
