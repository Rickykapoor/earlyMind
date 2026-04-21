# =============================================================================
# EarlyMind Dockerfile — Multi-stage CPU inference image
# Stage: base → inference
# Build:  docker build -t earlymind .
# Run:    docker run -p 8000:8000 -p 8501:8501 earlymind
# =============================================================================

FROM python:3.10-slim as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p checkpoints datasets/processed/eeg datasets/processed/mri datasets/processed/facial reports

COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/startup.sh"]
