#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNTIME_ENV="${SCRIPT_DIR}/runtime-env.sh"
if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Error: runtime-env.sh not found at ${RUNTIME_ENV}" >&2
  exit 1
fi
source "$RUNTIME_ENV"

CACHE_DIR="${MODEL_CACHE_DIR:-/mnt/disk2/shared-models}"
EPITOME_BACKEND_IMAGE="${EPITOME_BACKEND_IMAGE:-harbor.lji.org/iedb-intel/epitome-backend:latest}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --image)     EPITOME_BACKEND_IMAGE="$2"; shift 2 ;;
    *) break ;;
  esac
done

mkdir -p "$CACHE_DIR"

exec docker run --rm \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  -e MODEL="${MODEL}" \
  -e VLLM_HOST="${VLLM_HOST}" \
  -e VLLM_PORT="${VLLM_PORT}" \
  --cap-add=sys_nice \
  --ipc=host \
  -p "${VLLM_PORT}:${VLLM_PORT}" \
  -v "${CACHE_DIR}:/root/.cache" \
  -d \
  "${EPITOME_BACKEND_IMAGE}"
