#!/usr/bin/env bash
# nvidia-smi shim for AMD ROCm containers (rocm/pytorch base image).
# rocm-smi is available natively in this image so we use it directly.

ARGS="$*"

# nvidia-smi --query-gpu=index --format=csv,noheader
if echo "$ARGS" | grep -q -- "--query-gpu=index" && echo "$ARGS" | grep -q -- "csv,noheader"; then
    rocm-smi --showid --csv 2>/dev/null \
      | awk -F',' 'NR>1 && $1 ~ /^card/ { sub(/^card/, "", $1); print $1 }'
    exit 0
fi

# nvidia-smi --query-gpu=name --format=csv,noheader
if echo "$ARGS" | grep -q -- "--query-gpu=name" && echo "$ARGS" | grep -q -- "csv,noheader"; then
    rocm-smi --showproductname --csv 2>/dev/null \
      | awk -F',' 'NR>1 { print $2 }'
    exit 0
fi

# nvidia-smi --query-gpu=memory.total --format=csv,noheader
if echo "$ARGS" | grep -q -- "--query-gpu=memory.total" && echo "$ARGS" | grep -q -- "csv,noheader"; then
    rocm-smi --showmeminfo vram --csv 2>/dev/null \
      | awk -F',' 'NR>1 { printf "%.0f MiB\n", $2/1024/1024 }'
    exit 0
fi

# nvidia-smi -L
if echo "$ARGS" | grep -qE "(^| )-L( |$)"; then
    rocm-smi --showproductname --csv 2>/dev/null \
      | awk -F',' 'NR>1 { sub(/^card/, "", $1); print "GPU " $1 ": " $2 }'
    exit 0
fi

# Default — pass through to rocm-smi
rocm-smi "$@"
exit $?
