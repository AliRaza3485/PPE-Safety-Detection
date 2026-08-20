# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System libs needed by opencv-python-headless and torch on a slim base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install CPU-only torch/torchvision FIRST from the CPU wheel index — this must
# happen before requirements.txt is installed, otherwise pip may pull the much
# larger default (CUDA-enabled) torch build from PyPI.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# App code — copied after deps so code changes don't invalidate the dependency layer
COPY api/ ./api/
COPY src/ ./src/
COPY models/best.pt ./models/best.pt

# Make the src-layout "ppe_detector" package importable WITHOUT running
# `pip install .` (that would pull pyproject.toml's heavy training deps).
ENV PYTHONPATH=/app/src

ENV OMP_NUM_THREADS=1
ENV PPE_DEVICE=cpu
ENV PPE_WEIGHTS=models/best.pt

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]