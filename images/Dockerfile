FROM python:3.12-slim

RUN apt-get update -qq \
 && apt-get install -y -qq ripgrep \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    numpy \
    scipy \
    scikit-learn \
    matplotlib

# Pre-cache google/gemma-4-E2B-it weights and Wikitext-2.
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python - <<'EOF'
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
token = os.environ.get("HF_TOKEN") or None
print("Downloading google/gemma-4-E2B-it …")
snapshot_download("google/gemma-4-E2B-it", token=token)
print("Downloading wikitext-2-raw-v1 …")
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", token=token)
print("Pre-cache complete.")
EOF

# Ensure a clean, empty workspace.
RUN rm -rf /workspace && mkdir -p /workspace

COPY images/model_smoke.py /workspace/model_smoke.py
