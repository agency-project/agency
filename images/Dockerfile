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

# Ensure a clean, empty workspace.
RUN rm -rf /workspace && mkdir -p /workspace
