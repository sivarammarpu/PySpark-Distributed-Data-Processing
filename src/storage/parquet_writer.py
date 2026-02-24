"""
src/storage/parquet_writer.py
==============================
Parquet write and validation utilities.

Features
--------
- Snappy-compressed Parquet output
- Partition by one or more columns (date, region, category)
- Dynamic partition overwrite mode
- Post-write validation (row count + schema check)
"""

from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Write
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def write_parquet(
    df: DataFrame,
    output_path: str,
    partition_cols: Optional[List[str]] = None,
    mode: str = "overwrite",
    compression: str = "snappy",
) -> None:
    """
    Write a DataFrame to Parquet with optional column partitioning.

    Parameters
    ----------
    df             : DataFrame to persist.
    output_path    : Destination directory path (local or cloud URI).
    partition_cols : Columns to partition the output by.
                     E.g. ``["order_year", "order_month", "region"]``.
    mode           : ``overwrite``, ``append``, ``ignore``, or ``error``.
    compression    : ``snappy`` (default), ``gzip``, ``lz4``, ``none``.

    Notes
    -----
    Dynamic partition overwrite is controlled by
    ``spark.sql.sources.partitionOverwriteMode = dynamic``
    (set in ``spark_config.py``).  This means only the partitions present in
    ``df`` are overwritten — existing partitions not in ``df`` are untouched.
    """
    logger.info(
        "Writing Parquet → %s  [mode=%s  compression=%s  partitions=%s]",
        output_path, mode, compression, partition_cols,
    )

    writer = (
        df.write
        .format("parquet")
        .option("compression", compression)
        .mode(mode)
    )

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.save(output_path)
    logger.info("Parquet write complete → %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Read & Validate
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def read_and_validate_parquet(
    spark: SparkSession,
    path: str,
    expected_count: Optional[int] = None,
    expected_schema_cols: Optional[List[str]] = None,
    partition_filter: Optional[str] = None,
) -> DataFrame:
    """
    Read Parquet output and validate correctness.

    Parameters
    ----------
    spark                : Active SparkSession.
    path                 : Directory to read from.
    expected_count       : If provided, asserts row count == expected_count.
    expected_schema_cols : If provided, asserts all columns are present.
    partition_filter     : SQL predicate for partition pruning
                           e.g. ``"order_year = 2024 AND region = 'NORTH'"``.

    Returns
    -------
    DataFrame
    """
    logger.info("Reading Parquet from: %s", path)
    df = spark.read.parquet(path)

    if partition_filter:
        logger.info("Applying partition filter: %s", partition_filter)
        df = df.filter(partition_filter)

    actual_count = df.count()
    logger.info("Row count after read: %d", actual_count)

    # Column validation
    if expected_schema_cols:
        missing = [c for c in expected_schema_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet output missing expected columns: {missing}")
        logger.info("Schema validation passed — all expected columns present")

    # Row count validation
    if expected_count is not None and actual_count != expected_count:
        raise ValueError(
            f"Row count mismatch — expected: {expected_count}  actual: {actual_count}"
        )
    elif expected_count is not None:
        logger.info("Row count validation passed — %d rows", actual_count)

    df.printSchema()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Partition Listing Utility
# ═══════════════════════════════════════════════════════════════════════════

def list_partitions(spark: SparkSession, path: str) -> None:
    """Log all detected partitions under *path* for inspection."""
    try:
        df = spark.read.parquet(path)
        partitions = df.rdd.getNumPartitions()
        logger.info("Partition count in output: %d", partitions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list partitions at '%s': %s", path, exc)


# ═══════════════════════════════════════════════════════════════════════════
# ParquetStorage — convenience class
# ═══════════════════════════════════════════════════════════════════════════

class ParquetStorage:
    """
    Bundles Parquet write + validation into a single object.

    Usage
    -----
    >>> storage = ParquetStorage(spark, "data/output/sales/")
    >>> storage.write(df, partition_cols=["order_year", "region"])
    >>> validated_df = storage.validate(expected_count=50000)
    """

    def __init__(self, spark: SparkSession, output_path: str):
        self.spark = spark
        self.output_path = output_path

    def write(
        self,
        df: DataFrame,
        partition_cols: Optional[List[str]] = None,
        mode: str = "overwrite",
        compression: str = "snappy",
    ) -> None:
        write_parquet(df, self.output_path, partition_cols, mode, compression)

    def validate(
        self,
        expected_count: Optional[int] = None,
        expected_schema_cols: Optional[List[str]] = None,
        partition_filter: Optional[str] = None,
    ) -> DataFrame:
        return read_and_validate_parquet(
            self.spark,
            self.output_path,
            expected_count,
            expected_schema_cols,
            partition_filter,
        )
