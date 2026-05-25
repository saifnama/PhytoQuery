#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PhytoQuery — Qdrant Server (Docker)
# ─────────────────────────────────────────────────────────────────────────────
# Manage a single Qdrant container next to the FastAPI backend. Idempotent:
# re-running `start` when already running is a no-op, re-running on a stopped
# container resumes it without losing data.
#
# Usage:
#   ./scripts/qdrant.sh start      # start (creates container on first run)
#   ./scripts/qdrant.sh stop       # stop, keep state on disk
#   ./scripts/qdrant.sh status     # show container + health + collection count
#   ./scripts/qdrant.sh restart    # stop + start
#   ./scripts/qdrant.sh logs       # tail container logs
#   ./scripts/qdrant.sh remove     # stop + delete container; storage preserved
#
# All settings overridable via env vars (sensible defaults baked in):
#   QDRANT_CONTAINER    container name           default: phytoquery-qdrant
#   QDRANT_STORAGE_DIR  host storage path        default: ~/.local/share/phytoquery/qdrant_storage
#   QDRANT_VERSION      qdrant/qdrant image tag  default: v1.18.0
#                                                (matches qdrant-client in
#                                                 backend/requirements.txt)
#   QDRANT_PORT_REST    REST + Web UI port       default: 6333
#   QDRANT_PORT_GRPC    gRPC port                default: 6334
#
# Requirements:
#   * docker installed and runnable WITHOUT sudo (user in `docker` group
#     OR rootless docker daemon)
#   * curl (for health probe — optional, only used by `status` & `start`)
#
# Does NOT require sudo. Works identically on Linux and macOS.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NAME="${QDRANT_CONTAINER:-phytoquery-qdrant}"
STORAGE_DIR="${QDRANT_STORAGE_DIR:-${HOME}/.local/share/phytoquery/qdrant_storage}"
VERSION="${QDRANT_VERSION:-v1.18.0}"
PORT_REST="${QDRANT_PORT_REST:-6333}"
PORT_GRPC="${QDRANT_PORT_GRPC:-6334}"
IMAGE="qdrant/qdrant:${VERSION}"

REST_URL="http://localhost:${PORT_REST}"

# ─── prerequisite check ─────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found in PATH." >&2
    echo "  Install Docker (or Podman aliased as docker)." >&2
    exit 1
fi
if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: 'docker ps' failed. The Docker daemon is unreachable or" >&2
    echo "  your user can't talk to it. Either:" >&2
    echo "    (a) add your user to the docker group: sudo usermod -aG docker \$USER" >&2
    echo "        (then log out and back in), OR" >&2
    echo "    (b) use rootless docker: https://docs.docker.com/engine/security/rootless/" >&2
    exit 1
fi

# ─── helpers ────────────────────────────────────────────────────────────────
container_state() {
    # ``docker ps -a --filter name=^NAME$`` is preferred over
    # ``docker inspect --format '{{.State.Status}}'``:
    #   * Container-only lookup → no chance of matching an image,
    #     network, or volume with the same name (which on inspect
    #     would trigger "template parsing error: map has no entry
    #     for key 'State'").
    #   * Empty output means "no container" — cleaner than relying
    #     on inspect's non-zero exit code.
    #   * --format '{{.State}}' returns the state directly (running,
    #     exited, paused, created, restarting, dead).
    local line
    line=$(docker ps -a --filter "name=^${NAME}$" --format '{{.State}}' 2>/dev/null || true)
    if [[ -z "$line" ]]; then
        echo "absent"
    else
        echo "$line" | tr '[:upper:]' '[:lower:]'
    fi
}

print_urls() {
    echo "REST:    ${REST_URL}"
    echo "Web UI:  ${REST_URL}/dashboard"
    echo "gRPC:    localhost:${PORT_GRPC}"
    echo
    echo "Point PhytoQuery at it by setting (in .env / .env.<profile> or shell):"
    echo "  RAG_QDRANT_URL=${REST_URL}"
}

wait_for_health() {
    # Polls /healthz for up to 15s. Returns 0 on success, 1 on timeout.
    if ! command -v curl >/dev/null 2>&1; then
        echo "(curl not found; skipping health probe)"
        return 0
    fi
    for _ in $(seq 1 30); do
        if curl -fsS "${REST_URL}/healthz" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# ─── commands ───────────────────────────────────────────────────────────────
cmd_start() {
    state=$(container_state)
    case "$state" in
        running)
            echo "Already running."
            ;;
        exited|created|paused)
            echo "Starting existing container '${NAME}' (state: ${state})..."
            docker start "$NAME" >/dev/null
            ;;
        absent)
            mkdir -p "$STORAGE_DIR"
            echo "Creating container '${NAME}' from image ${IMAGE}..."
            echo "  Storage: ${STORAGE_DIR}"
            docker run -d \
                --name "$NAME" \
                -p "${PORT_REST}:6333" -p "${PORT_GRPC}:6334" \
                -v "${STORAGE_DIR}:/qdrant/storage" \
                --restart unless-stopped \
                "$IMAGE" >/dev/null
            ;;
        *)
            echo "ERROR: container in unexpected state '${state}'." >&2
            echo "Try: docker inspect ${NAME}" >&2
            exit 1
            ;;
    esac

    if wait_for_health; then
        echo "Qdrant is healthy."
    else
        echo "WARN: /healthz did not respond within 15s. Check 'docker logs ${NAME}'." >&2
    fi
    echo
    print_urls
}

cmd_stop() {
    state=$(container_state)
    if [[ "$state" == "running" ]]; then
        docker stop "$NAME" >/dev/null
        echo "Stopped (state preserved on disk at ${STORAGE_DIR})."
    else
        echo "Not running (state: ${state})."
    fi
}

cmd_status() {
    state=$(container_state)
    echo "Container: ${NAME}"
    echo "State:     ${state}"
    echo "Image:     ${IMAGE}"
    echo "Storage:   ${STORAGE_DIR}"
    if [[ "$state" == "running" ]]; then
        echo "Ports:     REST=${PORT_REST}  gRPC=${PORT_GRPC}"
        echo
        if command -v curl >/dev/null 2>&1; then
            if curl -fsS "${REST_URL}/healthz" >/dev/null 2>&1; then
                echo "Health:    OK (${REST_URL}/healthz)"
                if command -v python3 >/dev/null 2>&1; then
                    count=$(curl -fsS "${REST_URL}/collections" 2>/dev/null \
                            | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("result",{}).get("collections",[])))' \
                            2>/dev/null || echo "?")
                    echo "Collections: ${count}"
                fi
            else
                echo "Health:    UNREACHABLE — check 'docker logs ${NAME}'"
            fi
        fi
        echo
        print_urls
    fi
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_logs() {
    docker logs -f --tail 200 "$NAME"
}

cmd_remove() {
    cmd_stop
    if [[ "$(container_state)" != "absent" ]]; then
        docker rm "$NAME" >/dev/null
        echo "Container '${NAME}' removed."
    fi
    echo "Storage at ${STORAGE_DIR} preserved (delete manually if you want a clean slate)."
}

usage() {
    cat >&2 <<EOF
Usage: $0 {start|stop|status|restart|logs|remove}

  start    Start the Qdrant container (creates on first run; idempotent)
  stop     Stop the container; on-disk storage preserved
  status   Show container state, health, collection count, URLs
  restart  Stop + start
  logs     Tail container logs (Ctrl+C to exit)
  remove   Stop + delete container; on-disk storage preserved

Settings via env vars:
  QDRANT_CONTAINER     (default: phytoquery-qdrant)
  QDRANT_STORAGE_DIR   (default: \$HOME/.local/share/phytoquery/qdrant_storage)
  QDRANT_VERSION       (default: v1.18.0)
  QDRANT_PORT_REST     (default: 6333)
  QDRANT_PORT_GRPC     (default: 6334)
EOF
    exit 2
}

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_restart ;;
    logs)    cmd_logs ;;
    remove)  cmd_remove ;;
    -h|--help|help) usage ;;
    *)       usage ;;
esac
