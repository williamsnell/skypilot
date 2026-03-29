# Slurm Poll-Based Task Queue Integration Tests

Tests the poll-based task queue against a real Slurm cluster in containers.

## Quick Start (recommended)

Run everything inside the test runner container — fully self-contained:

```bash
# Build and start the Slurm cluster
podman compose -f tests/integration_tests/slurm_tunnel/docker-compose.yml up -d --build

# Run tests inside the test runner container
podman compose -f tests/integration_tests/slurm_tunnel/docker-compose.yml \
    run --rm --profile test testrunner \
    python -m pytest tests/integration_tests/slurm_tunnel/ -v

# Tear down
podman compose -f tests/integration_tests/slurm_tunnel/docker-compose.yml down -v
```

## Development (host-side)

For faster iteration, run tests on the host with the cluster in containers:

```bash
# Start the cluster
bash tests/integration_tests/slurm_tunnel/setup_keys.sh
podman compose -f tests/integration_tests/slurm_tunnel/docker-compose.yml up -d --build

# Start the API server
sky api stop && sky api start

# Run tests
python -m pytest tests/integration_tests/slurm_tunnel/ -v

# Tear down
podman compose -f tests/integration_tests/slurm_tunnel/docker-compose.yml down -v
```

## What's Tested

1. **Cluster connectivity** — SSH to slurmctld, sinfo
2. **Poll worker heartbeat** — worker connects and sends heartbeats to API server
3. **Task execution** — setup and run tasks delivered and executed via poll queue
4. **Signature verification** — tasks with invalid Ed25519 signatures are rejected
5. **Token authentication** — requests with invalid tokens are rejected

## Container Image

Slurm cluster: `giovtorres/slurm-docker-cluster` pinned by digest (see `Dockerfile.test`).
Test runner: `python:3.11-slim` with SkyPilot source mounted.
