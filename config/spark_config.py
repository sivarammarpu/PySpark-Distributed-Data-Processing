"""
config/spark_config.py
======================
Centralised SparkSession builder with production-tuned configurations.

Supports three execution environments:
  - local   : development on a single machine
  - staging : small cluster / test environment
  - prod    : full YARN / Kubernetes cluster
"""

from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# Default configuration values (override via environment variables or kwargs)
# ---------------------------------------------------------------------------
DEFAULT_CONFIGS = {
    # ── Memory & cores ──────────────────────────────────────────────────────
    "spark.executor.memory": "4g",
    "spark.driver.memory": "2g",
    "spark.executor.cores": "2",
    "spark.executor.instances": "4",

    # ── Serialisation ────────────────────────────────────────────────────────
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.kryoserializer.buffer.max": "512m",

    # ── Shuffle & partitions ─────────────────────────────────────────────────
    "spark.sql.shuffle.partitions": "200",
    "spark.default.parallelism": "200",

    # ── Adaptive Query Execution (Spark 3.x) ────────────────────────────────
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",

    # ── Dynamic Partition Overwrite ──────────────────────────────────────────
    "spark.sql.sources.partitionOverwriteMode": "dynamic",

    # ── Parquet optimisation ─────────────────────────────────────────────────
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.sql.parquet.mergeSchema": "false",
    "spark.sql.parquet.filterPushdown": "true",

    # ── Broadcast join threshold ─────────────────────────────────────────────
    "spark.sql.autoBroadcastJoinThreshold": "10485760",  # 10 MB

    # ── Network & I/O ────────────────────────────────────────────────────────
    "spark.network.timeout": "800s",
    "spark.executor.heartbeatInterval": "60s",
    "spark.sql.files.maxPartitionBytes": "134217728",  # 128 MB

    # ── Dynamic Allocation (YARN / K8s) ─────────────────────────────────────
    "spark.dynamicAllocation.enabled": "false",  # enable in prod
    "spark.dynamicAllocation.minExecutors": "2",
    "spark.dynamicAllocation.maxExecutors": "20",
    "spark.dynamicAllocation.initialExecutors": "4",
}


def build_spark_session(
    app_name: str = "DistributedDataPipeline",
    env: str = "local",
    extra_configs: dict = None,
) -> SparkSession:
    """
    Build and return a SparkSession with environment-appropriate settings.

    Parameters
    ----------
    app_name : str
        Human-readable name shown in the Spark UI.
    env : str
        Execution environment — one of ``local``, ``staging``, ``prod``.
    extra_configs : dict, optional
        Additional key-value Spark config pairs that override defaults.

    Returns
    -------
    SparkSession
    """
    builder = SparkSession.builder.appName(app_name)

    # ── Master URL ───────────────────────────────────────────────────────────
    if env == "local":
        # Use all available local cores; convenient for development
        builder = builder.master("local[*]")
    elif env == "staging":
        # Master is injected via spark-submit in staging/prod
        pass
    # prod: spark-submit handles master URL via --master flag

    # ── Apply default configs ────────────────────────────────────────────────
    for key, value in DEFAULT_CONFIGS.items():
        builder = builder.config(key, value)

    # ── Local-mode specific overrides ────────────────────────────────────────
    if env == "local":
        builder = (
            builder
            .config("spark.sql.shuffle.partitions", "8")   # lighter for local
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "2g")
        )

    # ── Override with caller-supplied configs ────────────────────────────────
    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, str(value))

    # ── Enable Hive support (needed for global temp views & metastore) ────────
    try:
        builder = builder.enableHiveSupport()
    except Exception:
        # Hive jars not on classpath in lightweight environments — skip silently
        pass

    spark = builder.getOrCreate()

    # ── Post-creation settings ────────────────────────────────────────────────
    spark.sparkContext.setLogLevel("WARN")  # suppress INFO noise in console

    return spark


def stop_spark_session(spark: SparkSession) -> None:
    """Gracefully stop the SparkSession."""
    if spark and not spark.sparkContext._jsc.sc().isStopped():
        spark.stop()
