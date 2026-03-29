#!/bin/bash
# Generate test SSH keys for the Slurm tunnel integration tests.
# These are TEST-ONLY keys, not used for any real authentication.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
KEY_DIR="$DIR/.test_keys"

if [ -f "$KEY_DIR/id_ed25519" ]; then
    echo "Test SSH keys already exist at $KEY_DIR"
    exit 0
fi

mkdir -p "$KEY_DIR"
ssh-keygen -t ed25519 -f "$KEY_DIR/id_ed25519" -N "" -q
echo "Test SSH keys generated at $KEY_DIR"
