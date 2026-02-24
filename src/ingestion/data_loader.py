"""
src/ingestion/data_loader.py
============================
All data-ingestion utilities for the distributed pipeline.

Supports:
  - CSV  (schema inference + manual schema, corrupt-record handling)
  - JSON (multiline, corrupt-record handling)
  - JDBC (relational DB — parameterised stub)
  - Cloud storage paths (S3 / Azure Blob / GCS)
  - DataFrame-level validation
  - Filter-based incremental loading
"""

from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from src.utils.logger import get_logger, log_execution_time
from src.utils.retry import retry

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CSV Ingestion
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
@retry(max_attempts=3, delay=2.0)
def load_csv(
    spark: SparkSession,
    path: str,
    schema: Optional[StructType] = None,
    header: bool = True,
    infer_schema: bool = True,
    mode: str = "PERMISSIVE",
    delimiter: str = ",",
) -> DataFrame:
    """
    Load a CSV file (or directory of CSV files) into a DataFrame.

    Parameters
    ----------
    spark        : Active SparkSession.
    path         : File path or glob pattern.
    schema       : Optional manual StructType.  If None, schema is inferred.
    header       : Whether the first row contains column names.
    infer_schema : Infer column data types when *schema* is None.
    mode         : Corrupt-record handling mode —
                   ``PERMISSIVE`` (default) stores bad rows in ``_corrupt_record``,
                   ``DROPMALFORMED`` silently drops them,
                   ``FAILFAST`` raises an exception immediately.
    delimiter    : Field delimiter character.

    Returns
    -------
    DataFrame
    """
    logger.info("Loading CSV from: %s  [mode=%s]", path, mode)

    reader = (
        spark.read
        .option("header", str(header).lower())
        .option("mode", mode)
        .option("sep", delimiter)
        .option("nullValue", "")
        .option("emptyValue", "")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .option("dateFormat", "yyyy-MM-dd")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
    )

    if schema:
        reader = reader.schema(schema)
    else:
        reader = reader.option("inferSchema", str(infer_schema).lower())

    df = reader.csv(path)
    logger.info("CSV loaded — rows (estimated): %d", df.count())
    return df


# ═══════════════════════════════════════════════════════════════════════════
# JSON Ingestion
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
@retry(max_attempts=3, delay=2.0)
def load_json(
    spark: SparkSession,
    path: str,
    schema: Optional[StructType] = None,
    multiline: bool = False,
    mode: str = "PERMISSIVE",
) -> DataFrame:
    """
    Load a JSON file (or directory) into a DataFrame.

    Parameters
    ----------
    spark     : Active SparkSession.
    path      : File path or glob pattern.
    schema    : Optional manual StructType.
    multiline : Set to True for pretty-printed / multi-line JSON documents.
    mode      : Corrupt-record handling mode (same options as load_csv).

    Returns
    -------
    DataFrame
    """
    logger.info("Loading JSON from: %s  [multiline=%s, mode=%s]", path, multiline, mode)

    reader = (
        spark.read
        .option("multiline", str(multiline).lower())
        .option("mode", mode)
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .option("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss")
    )

    if schema:
        reader = reader.schema(schema)

    df = reader.json(path)
    logger.info("JSON loaded — rows (estimated): %d", df.count())
    return df


# ═══════════════════════════════════════════════════════════════════════════
# JDBC Ingestion
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def load_jdbc(
    spark: SparkSession,
    jdbc_url: str,
    table: str,
    properties: Dict[str, str],
    partition_column: Optional[str] = None,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
    num_partitions: int = 10,
    query: Optional[str] = None,
) -> DataFrame:
    """
    Load data from a relational database via JDBC with optional parallelism.

    Parameters
    ----------
    spark            : Active SparkSession.
    jdbc_url         : JDBC connection URL  (e.g. ``jdbc:postgresql://host/db``).
    table            : Table name or a sub-query alias.
    properties       : Connection properties: ``{"user": ..., "password": ..., "driver": ...}``.
    partition_column : Numeric column used to split parallel reads.
    lower_bound      : Minimum value for partition_column.
    upper_bound      : Maximum value for partition_column.
    num_partitions   : Number of parallel JDBC tasks.
    query            : Custom SQL query (used instead of *table* when provided).

    Returns
    -------
    DataFrame
    """
    logger.info("Loading JDBC from: %s  table/query=%s", jdbc_url, query or table)

    reader = spark.read.format("jdbc").option("url", jdbc_url)

    if query:
        reader = reader.option("query", query)
    else:
        reader = reader.option("dbtable", table)

    # Parallel partition reads
    if partition_column and lower_bound is not None and upper_bound is not None:
        reader = (
            reader
            .option("partitionColumn", partition_column)
            .option("lowerBound", lower_bound)
            .option("upperBound", upper_bound)
            .option("numPartitions", num_partitions)
        )

    for k, v in properties.items():
        reader = reader.option(k, v)

    df = reader.load()
    logger.info("JDBC load complete")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Cloud Storage Ingestion (S3 / Azure Blob / GCS)
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def load_cloud(
    spark: SparkSession,
    path: str,
    file_format: str = "parquet",
    schema: Optional[StructType] = None,
    options: Optional[Dict[str, str]] = None,
) -> DataFrame:
    """
    Load data from cloud object storage using Spark's native connectors.

    Cloud credentials must be configured via ``spark-defaults.conf`` or
    environment variables (``AWS_ACCESS_KEY_ID``, ``AZURE_STORAGE_ACCOUNT``, etc.).

    Supported path formats:
      - S3   : ``s3a://bucket/prefix/``
      - Azure: ``wasbs://container@account.blob.core.windows.net/path/``
      - GCS  : ``gs://bucket/path/``

    Parameters
    ----------
    spark       : Active SparkSession.
    path        : Cloud storage path.
    file_format : File format — ``parquet``, ``csv``, ``json``, ``orc``.
    schema      : Optional StructType.
    options     : Additional reader options.

    Returns
    -------
    DataFrame
    """
    logger.info("Loading cloud data from: %s  [format=%s]", path, file_format)

    reader = spark.read.format(file_format)
    if schema:
        reader = reader.schema(schema)
    for k, v in (options or {}).items():
        reader = reader.option(k, v)

    df = reader.load(path)
    logger.info("Cloud load complete")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# DataFrame Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_dataframe(
    df: DataFrame,
    required_columns: List[str],
    null_threshold: float = 0.5,
    min_row_count: int = 1,
) -> DataFrame:
    """
    Validate an ingested DataFrame for structural and quality constraints.

    Checks performed
    ----------------
    1. All *required_columns* are present.
    2. No required column exceeds *null_threshold* null ratio.
    3. Total row count ≥ *min_row_count*.

    Parameters
    ----------
    df               : DataFrame to validate.
    required_columns : Column names that must exist.
    null_threshold   : Maximum allowed fraction of nulls per column (0.0–1.0).
    min_row_count    : Minimum acceptable number of rows.

    Returns
    -------
    DataFrame
        The original DataFrame (unchanged) — validation is side-effect only.

    Raises
    ------
    ValueError
        On any validation failure.
    """
    logger.info("Running DataFrame validation …")

    # 1. Column presence
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2. Row count
    total = df.count()
    if total < min_row_count:
        raise ValueError(
            f"DataFrame has {total} rows — minimum required: {min_row_count}"
        )

    # 3. Null ratio per column
    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in required_columns]
    ).collect()[0].asDict()

    for col_name, null_count in null_counts.items():
        ratio = null_count / total if total else 0
        if ratio > null_threshold:
            raise ValueError(
                f"Column '{col_name}' has {ratio:.1%} nulls — threshold: {null_threshold:.1%}"
            )
        logger.info("  %-30s  nulls: %d / %d  (%.1f%%)", col_name, null_count, total, ratio * 100)

    logger.info("Validation passed — %d rows, %d columns", total, len(df.columns))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Incremental Loading
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def load_incremental(
    spark: SparkSession,
    path: str,
    watermark_column: str,
    last_processed_value,
    file_format: str = "parquet",
    schema: Optional[StructType] = None,
) -> DataFrame:
    """
    Filter-based incremental load — returns only rows newer than *last_processed_value*.

    This is a simple high-watermark approach suitable for batch pipelines.
    For true streaming incremental loads, use Spark Structured Streaming.

    Parameters
    ----------
    spark                : Active SparkSession.
    path                 : Data source path.
    watermark_column     : Column used for incremental filtering (e.g., ``"updated_at"``).
    last_processed_value : Scalar value; only rows where watermark > this are returned.
    file_format          : File format to read.
    schema               : Optional StructType.

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only new/updated records.
    """
    logger.info(
        "Incremental load from '%s' — %s > %s",
        path, watermark_column, last_processed_value,
    )

    reader = spark.read.format(file_format)
    if schema:
        reader = reader.schema(schema)

    df = reader.load(path)
    df_new = df.filter(F.col(watermark_column) > last_processed_value)

    new_count = df_new.count()
    logger.info("Incremental rows loaded: %d", new_count)
    return df_new


# ═══════════════════════════════════════════════════════════════════════════
# Convenience loader dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def load_data(
    spark: SparkSession,
    path: str,
    file_format: str = "csv",
    schema: Optional[StructType] = None,
    **options,
) -> DataFrame:
    """
    Central dispatcher that routes to the correct loader based on *file_format*.

    Parameters
    ----------
    spark       : Active SparkSession.
    path        : Source path.
    file_format : One of ``csv``, ``json``, ``parquet``, ``orc``.
    schema      : Optional StructType.
    **options   : Forwarded to the underlying loader.

    Returns
    -------
    DataFrame
    """
    dispatch = {
        "csv":     load_csv,
        "json":    load_json,
        "parquet": load_cloud,
        "orc":     load_cloud,
    }
    loader = dispatch.get(file_format.lower())
    if not loader:
        raise ValueError(
            f"Unsupported format '{file_format}'.  "
            f"Choose from: {list(dispatch.keys())}"
        )
    return loader(spark, path, schema=schema, **options)
