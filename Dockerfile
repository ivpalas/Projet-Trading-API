# Dockerfile pour le Projet de Trading GBP/USD
# T12 - Containerisation

# ============================================================================
# BASE IMAGE
# ============================================================================
FROM python:3.10-slim

# Metadata
LABEL maintainer="Ivin Palas"
LABEL description="API de Trading GBP/USD avec ML et RL"
LABEL version="1.0"

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Working directory
WORKDIR /app

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# PYTHON DEPENDENCIES
# ============================================================================
# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# APPLICATION CODE
# ============================================================================
# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/saved \
    models/registry \
    data/processed \
    logs

# ============================================================================
# EXPOSE PORT
# ============================================================================
EXPOSE 8000

# ============================================================================
# HEALTH CHECK
# ============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# ============================================================================
# RUN APPLICATION
# ============================================================================
# Default command: Run FastAPI with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
