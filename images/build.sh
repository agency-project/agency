#!/usr/bin/env bash
# Build agency-sandbox:latest, auto-detecting the host GPU type.
# Pass GPU_TYPE=nvidia|rocm|cpu explicitly to override detection.
#
# Builds for every container runtime installed on the host, podman first --
# agsandbox_backend's "auto" backend selection prefers podman over docker
# when both are usable (see agsandbox_backend.get_container_runtime()), so
# an image built only for docker would leave podman's separate image store
# empty: podman would then try (and fail) to pull agency-sandbox from a
# registry instead of finding it locally.
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
    # `--secret id=...,env=VAR` is a docker buildx-only shorthand -- podman's
    # buildah rejects it ("incorrect secret flag format: should be
    # --secret id=foo,src=bar"). A file-based secret (src=) is the one form
    # both tools accept, so write the token to a private tempfile instead.
    HF_TOKEN_FILE="$(mktemp)"
    trap 'rm -f "$HF_TOKEN_FILE"' EXIT
    chmod 600 "$HF_TOKEN_FILE"
    printf '%s' "$HF_TOKEN" > "$HF_TOKEN_FILE"
    SECRET_FLAGS="--secret id=hf_token,src=${HF_TOKEN_FILE}"
fi

case "$GPU_TYPE" in
    rocm)   GPU_FLAGS="--device /dev/kfd --device /dev/dri" ;;
    nvidia) GPU_FLAGS="--gpus all" ;;
    *)      GPU_FLAGS="" ;;
esac

RUNTIMES=()
command -v podman &>/dev/null && RUNTIMES+=("podman")
command -v docker &>/dev/null && RUNTIMES+=("docker")

if [ "${#RUNTIMES[@]}" -eq 0 ]; then
    echo "Neither docker nor podman is installed." >&2
    exit 1
fi

for RUNTIME in "${RUNTIMES[@]}"; do
    # Podman requires a fully-qualified name (localhost/ prefix) to resolve
    # an unqualified image reference to its own local store instead of
    # searching configured remote registries; docker has no such requirement.
    if [ "$RUNTIME" = "podman" ]; then
        TAG="localhost/agency-sandbox:latest"
    else
        TAG="agency-sandbox:latest"
    fi

    echo "Building ${TAG} with ${RUNTIME} …"
    "$RUNTIME" build \
        --network=host \
        $SECRET_FLAGS \
        -t "$TAG" \
        -f "$DOCKERFILE" \
        "${REPO_ROOT}"

    echo "Running smoke test (${RUNTIME}) …"
    "$RUNTIME" run --rm $GPU_FLAGS "$TAG" python -c "import torch; print('torch', torch.__version__, 'OK')"
done
