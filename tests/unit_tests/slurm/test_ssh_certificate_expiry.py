"""Test that expired SSH certificates are detected before SSH is
attempted, with a clear error surfaced to the user.

Without this check, SSH with an expired certificate fails with a
cryptic error deep in the SSH connection. For Slurm clusters (e.g.,
Isambard) where certificates have short lifetimes, this is a common
failure mode — especially during sky down when certs may have expired
since launch.
"""
import os
import subprocess
import sys
import tempfile

import pytest

from sky import exceptions


def _make_expired_cert():
    """Create an expired SSH certificate for testing.

    Returns (private_key_path, cert_path) where the cert expired 1s ago.
    """
    tmpdir = tempfile.mkdtemp()
    ca_key = os.path.join(tmpdir, 'ca')
    user_key = os.path.join(tmpdir, 'user')

    # Generate CA key.
    subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', ca_key, '-N', ''],
                   check=True,
                   capture_output=True)
    # Generate user key.
    subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', user_key, '-N', ''],
                   check=True,
                   capture_output=True)
    # Sign with validity in the past (expired).
    subprocess.run([
        'ssh-keygen', '-s', ca_key, '-I', 'test-expired', '-V', '-1d:-1s',
        user_key + '.pub'
    ],
                   check=True,
                   capture_output=True)

    cert_path = user_key + '-cert.pub'
    assert os.path.exists(cert_path), f'Certificate not created at {cert_path}'
    return user_key, cert_path


def _make_valid_cert():
    """Create a valid SSH certificate for testing.

    Returns (private_key_path, cert_path) with 1 hour validity.
    """
    tmpdir = tempfile.mkdtemp()
    ca_key = os.path.join(tmpdir, 'ca')
    user_key = os.path.join(tmpdir, 'user')

    subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', ca_key, '-N', ''],
                   check=True,
                   capture_output=True)
    subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', user_key, '-N', ''],
                   check=True,
                   capture_output=True)
    subprocess.run([
        'ssh-keygen', '-s', ca_key, '-I', 'test-valid', '-V', '-5m:+1h',
        user_key + '.pub'
    ],
                   check=True,
                   capture_output=True)

    return user_key, user_key + '-cert.pub'


def test_expired_certificate_raises_before_ssh():
    """Attempting SSH with an expired certificate must raise a clear
    error before the SSH connection is attempted.

    The error must be a specific exception (not a generic SSH error)
    so callers like sky down can handle it appropriately.
    """
    from sky.utils.command_runner import SSHCommandRunner

    private_key, cert_path = _make_expired_cert()

    with pytest.raises(exceptions.SSHCertificateExpiredError):
        SSHCommandRunner(
            node=('127.0.0.1', 22),
            ssh_user='test',
            ssh_private_key=private_key,
            ssh_certificate_file=cert_path,
        )


def test_valid_certificate_does_not_raise():
    """A valid (non-expired) certificate must not raise."""
    from sky.utils.command_runner import SSHCommandRunner

    private_key, cert_path = _make_valid_cert()

    # Should not raise.
    runner = SSHCommandRunner(
        node=('127.0.0.1', 22),
        ssh_user='test',
        ssh_private_key=private_key,
        ssh_certificate_file=cert_path,
    )
    assert runner.ssh_certificate_file == cert_path


def test_error_survives_server_client_serialization():
    """SSHCertificateExpiredError must survive serialization from
    server to client.

    The API server serializes exceptions via sky.exceptions.serialize_exception
    and the client deserializes via deserialize_exception. If the error
    type isn't found in sky.exceptions globals, it degrades to a generic
    Exception and the user loses the clear error message and type.
    """
    original = exceptions.SSHCertificateExpiredError(
        'SSH certificate /path/to/cert expired at 2026-03-31T12:00:00. '
        'Please renew your certificate (e.g., run sky setup-slurm-ssh).')

    serialized = exceptions.serialize_exception(original)
    deserialized = exceptions.deserialize_exception(serialized)

    assert type(deserialized) is exceptions.SSHCertificateExpiredError, (
        f'Exception type lost during serialization: '
        f'got {type(deserialized).__name__}, expected '
        f'SSHCertificateExpiredError. The client will not see the '
        f'correct error type.')
    assert 'expired' in str(deserialized).lower()
    assert 'setup-slurm-ssh' in str(deserialized)


def test_no_certificate_does_not_raise():
    """SSH without a certificate file must not raise."""
    from sky.utils.command_runner import SSHCommandRunner

    tmpdir = tempfile.mkdtemp()
    key = os.path.join(tmpdir, 'key')
    subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', key, '-N', ''],
                   check=True,
                   capture_output=True)

    # Should not raise.
    SSHCommandRunner(
        node=('127.0.0.1', 22),
        ssh_user='test',
        ssh_private_key=key,
    )
