#!/bin/bash
set -e

echo "=== Installing SkyPilot (base only) ==="
cd /skypilot
# Base install only — no cloud SDKs. Provides fastapi, sqlalchemy, etc.
uv pip install --system -e "." --prerelease=allow
# Test deps (cryptography is already in base install via SkyPilot deps)
uv pip install --system pytest pytest-asyncio uvicorn httpx

# Set up Slurm SSH config for remote API server credential delivery tests.
# The key is named skypilot_test to satisfy the skypilot_* validation.
if [ -f /tmp/test_keys/id_ed25519 ]; then
    mkdir -p /root/.ssh /root/.sky/slurm
    cp /tmp/test_keys/id_ed25519 /root/.ssh/skypilot_test
    chmod 600 /root/.ssh/skypilot_test
    cat > /root/.sky/slurm/config << 'SLURM_CONFIG'
Host test-cluster
    Hostname slurmctld
    Port 22
    User root
    IdentityFile ~/.ssh/skypilot_test
    IdentitiesOnly yes
    ContainerRuntime podman-hpc
SLURM_CONFIG
fi

echo "=== Running tests ==="
exec "$@" -c /skypilot/tests/integration_tests/slurm_tunnel/pytest.ini
