#!/bin/bash
set -e

echo "=== Installing SkyPilot (base only) ==="
cd /skypilot
uv pip install --system -e "." --prerelease=allow

echo "=== Configuring SSH ==="
mkdir -p /root/.ssh
# Do NOT copy the private key here. The key path in ~/.sky/slurm/config
# points to a file that doesn't exist on the server. This forces the
# ephemeral credential delivery path: the client must send the key
# content in the launch request.

echo "=== Configuring Slurm ==="
mkdir -p /root/.sky/slurm
cat > /root/.sky/slurm/config << 'SLURM_CONFIG'
Host test-cluster
    Hostname slurmctld
    Port 22
    ContainerRuntime podman-hpc
SLURM_CONFIG

echo "=== Starting SkyPilot API server ==="
export SKYPILOT_DEV=1
exec python -m sky.server.server --host 0.0.0.0
