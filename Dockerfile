FROM nvcr.io/nvidia/pytorch:24.04-py3

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install runpod diffusers transformers accelerate omegaconf numpy pillow requests

# Try to install TensorRT - if it fails, we'll install it at runtime
RUN pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.11.0 || \
    echo "TensorRT installation failed, will retry at runtime"

# Create necessary directories
RUN mkdir -p /workspace/models/onnx_fp8 && \
    mkdir -p /workspace/models/engine_fp8 && \
    mkdir -p /tmp/output

# Copy handler
COPY handler.py /workspace/handler.py

# Alternative: Download TensorRT demo files directly
RUN cd /workspace && \
    wget https://github.com/NVIDIA/TensorRT/archive/refs/heads/release/10.11.zip && \
    unzip 10.11.zip && \
    mv TensorRT-release-10.11 TensorRT && \
    rm 10.11.zip

# Set working directory
WORKDIR /workspace

CMD ["python", "-u", "handler.py"]