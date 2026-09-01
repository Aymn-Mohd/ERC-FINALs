#!/bin/bash
# Bring up the ERC simulation container on a WSL2 host.
#
# Identical to up.sh but layers docker-compose.wsl.yml on top, which is what
# makes Gazebo render on the real GPU instead of falling back to llvmpipe.
# See docker-compose.wsl.yml for the full explanation.
#
# Usage:
#   ./docker/up-wsl.sh            start (or restart) the container
#   ./docker/up-wsl.sh --build    rebuild the image from scratch first
#
# On a native Linux host use ./docker/up.sh instead.
set -e
cd "$(dirname "$0")"

xhost +local:docker 2>/dev/null || echo "[up-wsl] warning: xhost unavailable (install x11-xserver-utils)"

docker compose down 2>/dev/null || true
docker container stop erc_sim 2>/dev/null || true
docker container rm erc_sim 2>/dev/null || true

COMPOSE="-f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.wsl.yml"

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[up-wsl] ERROR: nvidia-smi found no GPU. Check the Windows driver." >&2
    exit 1
fi

ARGS=()
BUILD=0
for arg in "$@"; do
    if [ "$arg" = "--build" ]; then BUILD=1; else ARGS+=("$arg"); fi
done

if [ "$BUILD" -eq 1 ]; then
    echo "[up-wsl] Pulling base image"
    docker pull osrf/ros:humble-desktop
    echo "[up-wsl] Rebuilding from scratch"
    docker compose $COMPOSE build --no-cache
fi

docker compose $COMPOSE up -d "${ARGS[@]}"
echo "[up-wsl] Container running. Attach with: ./docker/attach.sh"
