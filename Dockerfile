FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# The package needs its README and source during installation.
COPY requirements-dashboard.txt ./
RUN pip install -r requirements-dashboard.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps .

COPY app/ ./app/
COPY configs/ ./configs/
COPY reports/ ./reports/
COPY docs/ ./docs/
COPY data/outputs/ ./data/outputs/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
