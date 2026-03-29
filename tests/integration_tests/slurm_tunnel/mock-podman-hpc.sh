#!/bin/bash
# Mock podman-hpc for integration testing.
#
# Runs commands directly on the host instead of in a container.
# Exercises the full SkyPilot podman-hpc code path without needing
# actual container infrastructure.

CMD="$1"
shift

case "$CMD" in
    pull|migrate|rm)
        # No-op for container lifecycle commands
        exit 0
        ;;
    run)
        # podman-hpc run [flags] <image> <command...>
        # Skip flags until we hit the image name, then run the command.
        while [[ "$1" == --* ]]; do
            case "$1" in
                --name|--env)
                    # Flags with a value: skip flag + value
                    shift 2 ;;
                *)
                    # Flags without a value (--gpu, --replace, etc.)
                    shift ;;
            esac
        done
        # $1 is now the image name — skip it
        shift
        # Rest is the command
        exec "$@"
        ;;
    exec)
        # podman-hpc exec [-d] [-i] [--env KEY=VAL] <container_name> <command...>
        DETACH=false
        while [[ "$1" == -* ]]; do
            case "$1" in
                -d) DETACH=true; shift ;;
                -i) shift ;;
                --env) shift 2 ;;
                *) shift ;;
            esac
        done
        # $1 is container name — skip it
        shift
        if $DETACH; then
            "$@" &
        else
            exec "$@"
        fi
        ;;
    *)
        echo "mock-podman-hpc: unknown command '$CMD'" >&2
        exit 1
        ;;
esac
