#!/usr/bin/env bash
# deploy/submit_job.sh
# ====================
# spark-submit script for staging and production deployments.
#
# Usage
# -----
#   chmod +x deploy/submit_job.sh
#
#   # YARN cluster
#   MASTER=yarn INPUT=s3a://bucket/raw/ OUTPUT=s3a://bucket/processed/ \
#     bash deploy/submit_job.sh
#
#   # Kubernetes
#   MASTER=k8s://https://<k8s-api-server> bash deploy/submit_job.sh

set -euo pipefail

# ── Configuration (override via environment variables) ────────────────────
APP_NAME="${APP_NAME:-DistributedDataPipeline}"
MASTER="${MASTER:-yarn}"
ENV="${ENV:-staging}"
INPUT="${INPUT:-s3a://my-bucket/raw/sales_data.csv}"
OUTPUT="${OUTPUT:-s3a://my-bucket/processed/}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-4g}"
DRIVER_MEMORY="${DRIVER_MEMORY:-2g}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
NUM_EXECUTORS="${NUM_EXECUTORS:-8}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-200}"
PYTHON_FILES="${PYTHON_FILES:-dist/pipeline.zip}"

echo "============================================================"
echo " Submitting: ${APP_NAME}"
echo " Master    : ${MASTER}"
echo " Env       : ${ENV}"
echo " Input     : ${INPUT}"
echo " Output    : ${OUTPUT}"
echo "============================================================"

spark-submit \
  --master              "${MASTER}" \
  --deploy-mode         cluster \
  --name                "${APP_NAME}" \
  --conf "spark.executor.memory=${EXECUTOR_MEMORY}" \
  --conf "spark.driver.memory=${DRIVER_MEMORY}" \
  --conf "spark.executor.cores=${EXECUTOR_CORES}" \
  --conf "spark.executor.instances=${NUM_EXECUTORS}" \
  --conf "spark.sql.shuffle.partitions=${SHUFFLE_PARTITIONS}" \
  --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
  --conf "spark.sql.adaptive.enabled=true" \
  --conf "spark.sql.adaptive.coalescePartitions.enabled=true" \
  --conf "spark.sql.sources.partitionOverwriteMode=dynamic" \
  --conf "spark.dynamicAllocation.enabled=true" \
  --conf "spark.dynamicAllocation.minExecutors=2" \
  --conf "spark.dynamicAllocation.maxExecutors=20" \
  --py-files             "${PYTHON_FILES}" \
  main.py \
    --input   "${INPUT}" \
    --output  "${OUTPUT}" \
    --env     "${ENV}" \
    --app-name "${APP_NAME}"

echo "Job submitted successfully."
