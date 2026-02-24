# deploy/docker/Dockerfile
# =========================
# Multi-stage Dockerfile for containerised PySpark execution.
# Produces a lean image suitable for Kubernetes (spark-on-k8s) or local testing.

# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# ── Java 11 (required by PySpark) ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── Python dependencies ───────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY . .

# ── Environment defaults ──────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# ── Entrypoint ────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]
CMD ["--env", "local", "--input", "data/raw/sales_data.csv", "--output", "data/output/"]
