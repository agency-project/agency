#!/usr/bin/env bash
# Build agency-sandbox:latest, auto-detecting the host GPU type.
# Pass GPU_TYPE=nvidia|rocm|cpu explicitly to override detection.
set -euo pipefail

if [ -z "${GPU_TYPE:-}" ]; then
    if nvidia-smi --query-gpu=index --format=csv,noheader &>/dev/null; then
        GPU_TYPE=nvidia
    elif rocm-smi --showid --csv &>/dev/null; then
        GPU_TYPE=rocm
    else
        GPU_TYPE=cpu
    fi
fi

echo "Building agency-sandbox:latest with GPU_TYPE=${GPU_TYPE}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$GPU_TYPE" in
    rocm)   DOCKERFILE="${REPO_ROOT}/images/Dockerfile.rocm" ;;
    nvidia) DOCKERFILE="${REPO_ROOT}/images/Dockerfile.nvidia" ;;
    *)      DOCKERFILE="${REPO_ROOT}/images/Dockerfile" ;;
esac

SECRET_FLAGS=""
if [ -n "${HF_TOKEN:-}" ]; then
    SECRET_FLAGS="--secret id=hf_token,env=HF_TOKEN"
fi

docker build \
    --network=host \
    $SECRET_FLAGS \
    -t agency-sandbox:latest \
    -f "$DOCKERFILE" \
    "${REPO_ROOT}"

echo "Running smoke test …"
case "$GPU_TYPE" in
    rocm)   GPU_FLAGS="--device /dev/kfd --device /dev/dri" ;;
    nvidia) GPU_FLAGS="--gpus all" ;;
    *)      GPU_FLAGS="" ;;
esac

docker run --rm $GPU_FLAGS agency-sandbox:latest python /opt/model_smoke.py
