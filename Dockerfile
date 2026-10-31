# Reproducible execution environment for RadNet Anomaly Detection Benchmark
FROM python:3.11-slim

# System dependencies for OpenMP (LightGBM) and basic build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install pinned Python dependencies
COPY requirements-lock.txt /workspace/
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy repository code
COPY . /workspace/

ENV PYTHONPATH=/workspace
ENV PYTHONUTF8=1

# Default command runs lightweight unit tests and verification
CMD ["python", "-m", "pytest", "-q", "-m", "not integration"]
