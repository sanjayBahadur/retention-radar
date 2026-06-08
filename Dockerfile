# Retention Radar Production Dockerfile
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install Python package dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install .

# Copy application source code and precomputed model artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY reports/ ./reports/
COPY bi/ ./bi/
COPY docs/ ./docs/
COPY data/outputs/ ./data/outputs/
COPY .streamlit/ ./.streamlit/

# Install local package in editable mode
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit Executive Command Center
ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
