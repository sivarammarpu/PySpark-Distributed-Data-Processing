"""
src/optimization/optimizer.py
==============================
Performance optimisation utilities for the distributed pipeline.

Strategies
----------
- Smart repartition vs coalesce
- Broadcast join hints
- DataFrame caching with configurable StorageLevels
- Partition skew diagnostics
- Salted join for skew mitigation
"""

from typing import Optional

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


# ── Storage level lookup ───────────────────────────────────────────────────
_STORAGE_LEVELS = {
    "memory_only":     StorageLevel.MEMORY_ONLY,
    "memory_and_disk": StorageLevel.MEMORY_AND_DISK,
    "disk_only":       StorageLevel.DISK_ONLY,
    "off_heap":        StorageLevel.OFF_HEAP,
}


# ═══════════════════════════════════════════════════════════════════════════
# Repartition vs Coalesce
# ═══════════════════════════════════════════════════════════════════════════

def smart_repartition(
    df: DataFrame,
    target: int,
    col: Optional[str] = None,
) -> DataFrame:
    """
    Choose coalesce (when reducing) or repartition (when increasing) to avoid
    unnecessary shuffles.  If *col* is given, always hash-repartitions by that column.
    """
    current = df.rdd.getNumPartitions()
    logger.info("Partitions — current: %d  target: %d  col: %s", current, target, col or "none")

    if col:
        df = df.repartition(target, F.col(col))
        logger.info("repartition(%d, %s)", target, col)
    elif current > target:
        df = df.coalesce(target)
        logger.info("coalesce(%d) — no full shuffle", target)
    else:
        df = df.repartition(target)
        logger.info("repartition(%d)", target)

    logger.info("New partition count: %d", df.rdd.getNumPartitions())
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Broadcast Join
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def broadcast_join(
    df_large: DataFrame,
    df_small: DataFrame,
    join_col: str,
    join_type: str = "inner",
) -> DataFrame:
    """
    Join ``df_large`` with a broadcast-hinted ``df_small``.
    Eliminates the shuffle stage on the large DataFrame.
    """
    logger.info("Broadcast join — col: %s  type: %s", join_col, join_type)
    result = df_large.join(F.broadcast(df_small), join_col, join_type)
    logger.info("Broadcast join complete")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Caching & Persistence
# ═══════════════════════════════════════════════════════════════════════════

def cache_dataframe(
    df: DataFrame,
    storage_level: str = "memory_and_disk",
    name: Optional[str] = None,
) -> DataFrame:
    """Persist a DataFrame with the chosen StorageLevel."""
    level = _STORAGE_LEVELS.get(storage_level, StorageLevel.MEMORY_AND_DISK)
    logger.info("Persisting '%s' — level: %s", name or "DF", storage_level)
    df.persist(level)
    return df


def unpersist_dataframe(df: DataFrame, name: Optional[str] = None) -> None:
    """Release a cached DataFrame from memory/disk."""
    df.unpersist()
    logger.info("Unpersisted '%s'", name or "DF")


# ═══════════════════════════════════════════════════════════════════════════
# Partition Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def log_partition_info(df: DataFrame, label: str = "") -> None:
    """Log partition count and per-partition row distribution for skew detection."""
    n = df.rdd.getNumPartitions()
    counts = [
        c for _, c in df.rdd.mapPartitionsWithIndex(
            lambda i, it: [(i, sum(1 for _ in it))]
        ).collect()
    ]
    total = sum(counts)
    mn, mx = (min(counts), max(counts)) if counts else (0, 0)
    avg = total / n if n else 0
    skew = (mx - mn) / avg if avg else 0

    logger.info(
        "[Partitions] %s — n=%d  rows=%d  min/max=%d/%d  skew=%.2fx",
        label or "DF", n, total, mn, mx, skew,
    )
    if skew > 2.0:
        logger.warning("High skew (%.2fx) — consider salting or repartition.", skew)


# ═══════════════════════════════════════════════════════════════════════════
# Skew Mitigation — Salting
# ═══════════════════════════════════════════════════════════════════════════

def salt_join(
    df_large: DataFrame,
    df_small: DataFrame,
    join_col: str,
    salt_buckets: int = 10,
) -> DataFrame:
    """
    Mitigate join skew via key salting.
    Adds random buckets to df_large keys and explodes equivalent copies in df_small.
    """
    logger.info("Salt join — col: %s  buckets: %d", join_col, salt_buckets)

    df_large = (
        df_large
        .withColumn("_salt", (F.rand() * salt_buckets).cast("int"))
        .withColumn(
            f"{join_col}_salted",
            F.concat(F.col(join_col).cast("string"), F.lit("_"), F.col("_salt")),
        )
    )
    df_small = (
        df_small
        .withColumn("_salt_arr", F.array([F.lit(i) for i in range(salt_buckets)]))
        .withColumn("_salt", F.explode("_salt_arr"))
        .withColumn(
            f"{join_col}_salted",
            F.concat(F.col(join_col).cast("string"), F.lit("_"), F.col("_salt")),
        )
        .drop("_salt_arr", "_salt")
    )

    result = df_large.join(df_small, f"{join_col}_salted").drop(f"{join_col}_salted", "_salt")
    logger.info("Salt join complete")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# DataOptimizer — builder class
# ═══════════════════════════════════════════════════════════════════════════

class DataOptimizer:
    """Fluent wrapper for all optimisation utilities."""

    def __init__(self, df: DataFrame):
        self._df = df
        self._cached = False

    def repartition(self, target: int, col: Optional[str] = None) -> "DataOptimizer":
        self._df = smart_repartition(self._df, target, col)
        return self

    def cache(self, storage_level: str = "memory_and_disk") -> "DataOptimizer":
        self._df = cache_dataframe(self._df, storage_level)
        self._cached = True
        return self

    def diagnose(self, label: str = "") -> "DataOptimizer":
        log_partition_info(self._df, label)
        return self

    def result(self) -> DataFrame:
        return self._df

    def cleanup(self) -> None:
        if self._cached:
            unpersist_dataframe(self._df)
            self._cached = False
