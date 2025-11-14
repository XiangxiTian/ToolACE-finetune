# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    tmux \
    nvtop \
    coreutils \
    procps \
    curl \
    ca-certificates \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements and install main dependencies globally using uv
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create isolated venv for quantization using uv (much faster!)
COPY requirements-quantize.txt /tmp/requirements-quantize.txt
RUN uv venv /opt/quantize_venv && \
    uv pip install --python /opt/quantize_venv/bin/python -r /tmp/requirements-quantize.txt && \
    rm /tmp/requirements-quantize.txt

# Create convenience alias for quantization venv activation
RUN echo 'alias activate-quantize="source /opt/quantize_venv/bin/activate"' >> /etc/bash.bashrc

# Set default command
CMD ["/bin/bash"]