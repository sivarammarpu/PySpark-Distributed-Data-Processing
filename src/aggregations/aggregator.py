"""
src/aggregations/aggregator.py
===============================
Distributed aggregation pipelines for analytical insights.

Covers:
  - GroupBy with multi-column aggregations
  - Window functions (row_number, rank, dense_rank, lag, lead)
  - Statistical aggregates (mean, stddev, min/max, percentile_approx)
  - Rolling / cumulative aggregations
"""

from typing import Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GroupBy Aggregations
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def group_aggregate(
    df: DataFrame,
    group_cols: List[str],
    agg_map: Dict[str, List[str]],
) -> DataFrame:
    """
    Perform GroupBy aggregation with multiple metrics per column.

    Parameters
    ----------
    df         : Source DataFrame.
    group_cols : Columns to group by.
    agg_map    : ``{column: [functions…]}`` e.g.
                 ``{"revenue": ["sum", "avg", "max"], "quantity": ["sum", "count"]}``.

    Returns
    -------
    DataFrame   — one row per group, one column per (col, func) combination.

    Example
    -------
    >>> group_aggregate(df, ["region", "order_year"],
    ...                 {"revenue": ["sum", "avg"], "quantity": ["sum"]})
    """
    logger.info("GroupBy: %s  metrics: %s", group_cols, agg_map)

    fn_map = {
        "sum":   F.sum,
        "avg":   F.avg,
        "mean":  F.avg,
        "max":   F.max,
        "min":   F.min,
        "count": F.count,
        "stddev": F.stddev,
    }

    agg_exprs = []
    for col_name, funcs in agg_map.items():
        for func_name in funcs:
            fn = fn_map.get(func_name.lower())
            if fn:
                agg_exprs.append(
                    fn(col_name).alias(f"{func_name}_{col_name}")
                )

    result = df.groupBy(*group_cols).agg(*agg_exprs)
    logger.info("GroupBy complete — result rows: %d", result.count())
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Window Functions
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def apply_window_functions(
    df: DataFrame,
    partition_col: str,
    order_col: str,
    value_col: str = "revenue",
    lag_periods: int = 1,
    lead_periods: int = 1,
) -> DataFrame:
    """
    Enrich a DataFrame with ranking and offset window functions.

    Columns added
    -------------
    * ``row_num``        — unique sequential row number within partition
    * ``rank``           — rank with gaps for ties
    * ``dense_rank``     — rank without gaps for ties
    * ``lag_{value_col}``  — value from *lag_periods* rows behind
    * ``lead_{value_col}`` — value from *lead_periods* rows ahead
    * ``running_total``  — cumulative sum of *value_col* within partition

    Parameters
    ----------
    df            : Source DataFrame.
    partition_col : Column used to define window partitions (e.g. ``"region"``).
    order_col     : Column used to order rows within each partition.
    value_col     : Column over which numeric window ops are applied.
    lag_periods   : Offset for lag function.
    lead_periods  : Offset for lead function.

    Returns
    -------
    DataFrame
    """
    logger.info(
        "Applying window functions — partition: %s, order: %s, value: %s",
        partition_col, order_col, value_col,
    )

    # Window specs ─────────────────────────────────────────────────────────
    w_rank    = Window.partitionBy(partition_col).orderBy(F.desc(order_col))
    w_running = Window.partitionBy(partition_col).orderBy(order_col).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )

    df = (
        df
        .withColumn("row_num",    F.row_number().over(w_rank))
        .withColumn("rank",       F.rank().over(w_rank))
        .withColumn("dense_rank", F.dense_rank().over(w_rank))
        .withColumn(f"lag_{value_col}",  F.lag(value_col, lag_periods).over(w_rank))
        .withColumn(f"lead_{value_col}", F.lead(value_col, lead_periods).over(w_rank))
        .withColumn("running_total", F.sum(value_col).over(w_running))
    )
    logger.info("Window functions applied")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Aggregations
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def statistical_aggregates(
    df: DataFrame,
    numeric_cols: List[str],
    percentiles: Optional[List[float]] = None,
) -> DataFrame:
    """
    Compute descriptive statistics for a list of numeric columns.

    Metrics computed per column
    ---------------------------
    mean, stddev, min, max, and optional percentile_approx values.

    Parameters
    ----------
    df           : Source DataFrame.
    numeric_cols : Columns to compute stats for.
    percentiles  : List of percentile values e.g. ``[0.25, 0.5, 0.75]``.
                   Omit for mean/stddev/min/max only.

    Returns
    -------
    DataFrame   — one row with one column per metric.
    """
    logger.info("Computing statistical aggregates for: %s", numeric_cols)

    if not percentiles:
        percentiles = []

    agg_exprs = []
    for col_name in numeric_cols:
        if col_name not in df.columns:
            logger.warning("Column '%s' not found — skipped", col_name)
            continue
        agg_exprs.extend([
            F.round(F.mean(col_name),   4).alias(f"mean_{col_name}"),
            F.round(F.stddev(col_name), 4).alias(f"stddev_{col_name}"),
            F.min(col_name).alias(f"min_{col_name}"),
            F.max(col_name).alias(f"max_{col_name}"),
        ])
        for p in percentiles:
            label = str(int(p * 100))
            agg_exprs.append(
                F.percentile_approx(col_name, p).alias(f"p{label}_{col_name}")
            )

    if not agg_exprs:
        logger.warning("No valid numeric columns found for statistical aggregation")
        return df.limit(0)

    result = df.agg(*agg_exprs)
    logger.info("Statistical aggregation complete")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Sales Summary (Domain-Specific Aggregation)
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def sales_summary(df: DataFrame) -> DataFrame:
    """
    Produce a rich sales summary aggregated by region and product.

    Included metrics
    ----------------
    total_revenue, total_orders, avg_order_value, max_revenue, unique_customers
    """
    logger.info("Building sales summary …")

    required = {"region", "revenue", "order_id", "customer_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sales_summary requires columns: {missing}")

    summary = df.groupBy("region", "product").agg(
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.count("order_id").alias("total_orders"),
        F.round(F.avg("revenue"), 2).alias("avg_order_value"),
        F.max("revenue").alias("max_revenue"),
        F.countDistinct("customer_id").alias("unique_customers"),
    ).orderBy(F.desc("total_revenue"))

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# DataAggregator — orchestration class
# ═══════════════════════════════════════════════════════════════════════════

class DataAggregator:
    """
    Convenience class that bundles all aggregation functions.

    Usage
    -----
    >>> agg = DataAggregator(df)
    >>> summary = agg.group(["region"], {"revenue": ["sum", "avg"]})
    >>> windowed = agg.window("region", "order_date", "revenue")
    >>> stats = agg.stats(["revenue", "quantity"])
    """

    def __init__(self, df: DataFrame):
        self._df = df

    def group(
        self,
        group_cols: List[str],
        agg_map: Dict[str, List[str]],
    ) -> DataFrame:
        return group_aggregate(self._df, group_cols, agg_map)

    def window(
        self,
        partition_col: str,
        order_col: str,
        value_col: str = "revenue",
    ) -> DataFrame:
        return apply_window_functions(self._df, partition_col, order_col, value_col)

    def stats(
        self,
        numeric_cols: List[str],
        percentiles: Optional[List[float]] = None,
    ) -> DataFrame:
        return statistical_aggregates(self._df, numeric_cols, percentiles)

    def summary(self) -> DataFrame:
        return sales_summary(self._df)
