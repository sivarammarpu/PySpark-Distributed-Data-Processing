"""
src/transformations/transformer.py
===================================
Distributed DataFrame transformation operations.

All functions are pure (return new DataFrames) and composable via the
``DataTransformer`` class builder pattern.
"""

from typing import Dict, List, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DataType

from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Individual transformation functions
# ═══════════════════════════════════════════════════════════════════════════

def filter_records(df: DataFrame, condition: str) -> DataFrame:
    """
    Filter rows using a SQL-style boolean string expression.

    Example
    -------
    >>> filter_records(df, "quantity > 0 AND unit_price IS NOT NULL")
    """
    logger.info("Filtering rows — condition: %s", condition)
    result = df.filter(condition)
    logger.info("Rows after filter: %d", result.count())
    return result


def select_rename(
    df: DataFrame,
    col_map: Dict[str, str],
) -> DataFrame:
    """
    Select a subset of columns and optionally rename them.

    Parameters
    ----------
    col_map : ``{original_name: new_name}`` mapping.
              If the names are identical, no rename occurs.

    Example
    -------
    >>> select_rename(df, {"order_id": "id", "unit_price": "price"})
    """
    logger.info("Selecting & renaming columns: %s", list(col_map.keys()))
    selected = [F.col(orig).alias(alias) for orig, alias in col_map.items()]
    return df.select(*selected)


def cast_columns(
    df: DataFrame,
    cast_map: Dict[str, Union[str, DataType]],
) -> DataFrame:
    """
    Cast columns to new data types.

    Parameters
    ----------
    cast_map : ``{column_name: target_type}`` mapping.
               Target type can be a string (``"double"``) or a StructType.

    Example
    -------
    >>> cast_columns(df, {"quantity": "integer", "unit_price": "double"})
    """
    logger.info("Casting columns: %s", cast_map)
    for col_name, target_type in cast_map.items():
        df = df.withColumn(col_name, F.col(col_name).cast(target_type))
    return df


def handle_nulls(
    df: DataFrame,
    strategy: str = "fill",
    fill_map: Optional[Dict[str, object]] = None,
    subset: Optional[List[str]] = None,
) -> DataFrame:
    """
    Handle null / NaN values across the DataFrame.

    Parameters
    ----------
    strategy : ``"fill"`` — replace nulls using *fill_map*.
               ``"drop"`` — drop rows that contain any null in *subset*.
               ``"fill_default"`` — fill numeric nulls with 0 and string nulls with "UNKNOWN".
    fill_map : ``{column_name: fill_value}`` used when strategy is ``"fill"``.
    subset   : Column list scope for ``"drop"`` strategy.

    Example
    -------
    >>> handle_nulls(df, strategy="fill", fill_map={"discount": 0.0, "region": "UNKNOWN"})
    """
    logger.info("Handling nulls — strategy: %s", strategy)
    if strategy == "drop":
        return df.dropna(subset=subset)
    elif strategy == "fill" and fill_map:
        return df.fillna(fill_map)
    elif strategy == "fill_default":
        numeric_cols = [
            f.name for f in df.schema.fields
            if str(f.dataType) in ("IntegerType()", "LongType()", "DoubleType()", "FloatType()")
        ]
        string_cols = [
            f.name for f in df.schema.fields
            if str(f.dataType) == "StringType()"
        ]
        df = df.fillna(0, subset=numeric_cols)
        df = df.fillna("UNKNOWN", subset=string_cols)
        return df
    else:
        logger.warning("No null strategy applied — check strategy/fill_map arguments.")
        return df


def add_derived_columns(df: DataFrame) -> DataFrame:
    """
    Add business-logic derived columns to a sales-style DataFrame.

    Columns added
    -------------
    * ``revenue``        — quantity × unit_price × (1 − discount)
    * ``profit_margin``  — revenue / unit_price (proxy margin %)
    * ``age_bucket``     — customer-age bucket (if ``customer_age`` present)
    * ``order_year``     — extracted from ``order_date``
    * ``order_month``    — extracted from ``order_date``
    * ``order_quarter``  — Q1–Q4 from ``order_date``
    * ``ingestion_ts``   — current timestamp when the record was processed
    """
    logger.info("Adding derived columns …")

    if "quantity" in df.columns and "unit_price" in df.columns:
        discount = F.col("discount") if "discount" in df.columns else F.lit(0.0)
        df = df.withColumn(
            "revenue",
            F.round(F.col("quantity") * F.col("unit_price") * (F.lit(1.0) - discount), 2),
        )
        df = df.withColumn(
            "profit_margin",
            F.when(F.col("unit_price") > 0,
                   F.round(F.col("revenue") / (F.col("quantity") * F.col("unit_price")), 4))
            .otherwise(F.lit(0.0)),
        )

    if "order_date" in df.columns:
        df = (
            df
            .withColumn("order_year",    F.year("order_date"))
            .withColumn("order_month",   F.month("order_date"))
            .withColumn("order_quarter", F.quarter("order_date"))
        )

    if "customer_age" in df.columns:
        df = df.withColumn(
            "age_bucket",
            F.when(F.col("customer_age") < 25, "18-24")
            .when(F.col("customer_age") < 35, "25-34")
            .when(F.col("customer_age") < 45, "35-44")
            .when(F.col("customer_age") < 55, "45-54")
            .otherwise("55+"),
        )

    df = df.withColumn("ingestion_ts", F.current_timestamp())
    logger.info("Derived columns added successfully")
    return df


def apply_conditional(df: DataFrame) -> DataFrame:
    """
    Apply conditional transformations using ``when / otherwise``.

    Transformations
    ---------------
    * ``sales_tier``     — Bronze / Silver / Gold / Platinum based on revenue
    * ``payment_flag``   — 1 / 0 online-payment indicator
    * ``discount_level`` — None / Low / Medium / High based on discount value
    """
    logger.info("Applying conditional transformations …")

    if "revenue" in df.columns:
        df = df.withColumn(
            "sales_tier",
            F.when(F.col("revenue") >= 10_000, "Platinum")
            .when(F.col("revenue") >= 5_000,  "Gold")
            .when(F.col("revenue") >= 1_000,  "Silver")
            .otherwise("Bronze"),
        )

    if "payment_method" in df.columns:
        df = df.withColumn(
            "payment_flag",
            F.when(
                F.col("payment_method").isin("Credit Card", "Digital Wallet"), 1
            ).otherwise(0),
        )

    if "discount" in df.columns:
        df = df.withColumn(
            "discount_level",
            F.when(F.col("discount") == 0,          "None")
            .when(F.col("discount") <= 0.10,         "Low")
            .when(F.col("discount") <= 0.25,         "Medium")
            .otherwise("High"),
        )

    return df


def drop_duplicate_records(
    df: DataFrame,
    subset: Optional[List[str]] = None,
) -> DataFrame:
    """Drop duplicate rows, optionally scoped to a subset of columns."""
    before = df.count()
    df = df.dropDuplicates(subset)
    after = df.count()
    logger.info("Duplicates removed: %d  (before=%d, after=%d)", before - after, before, after)
    return df


def standardise_strings(df: DataFrame, columns: List[str]) -> DataFrame:
    """Strip whitespace and upper-case the given string columns."""
    for col_name in columns:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.upper(F.trim(F.col(col_name))),
            )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# DataTransformer — composable builder class
# ═══════════════════════════════════════════════════════════════════════════

class DataTransformer:
    """
    Builder-style transformer that chains all transformation steps.

    Usage
    -----
    >>> transformer = DataTransformer(df)
    >>> result = (
    ...     transformer
    ...     .filter("quantity > 0")
    ...     .cast({"quantity": "integer", "unit_price": "double"})
    ...     .nulls(strategy="fill", fill_map={"discount": 0.0})
    ...     .derive()
    ...     .conditional()
    ...     .dedup(["order_id"])
    ...     .result()
    ... )
    """

    def __init__(self, df: DataFrame):
        self._df = df

    def filter(self, condition: str) -> "DataTransformer":
        self._df = filter_records(self._df, condition)
        return self

    def select(self, col_map: Dict[str, str]) -> "DataTransformer":
        self._df = select_rename(self._df, col_map)
        return self

    def cast(self, cast_map: Dict[str, Union[str, DataType]]) -> "DataTransformer":
        self._df = cast_columns(self._df, cast_map)
        return self

    def nulls(self, strategy: str = "fill", fill_map: Optional[Dict] = None,
              subset: Optional[List[str]] = None) -> "DataTransformer":
        self._df = handle_nulls(self._df, strategy, fill_map, subset)
        return self

    def derive(self) -> "DataTransformer":
        self._df = add_derived_columns(self._df)
        return self

    def conditional(self) -> "DataTransformer":
        self._df = apply_conditional(self._df)
        return self

    def dedup(self, subset: Optional[List[str]] = None) -> "DataTransformer":
        self._df = drop_duplicate_records(self._df, subset)
        return self

    def standardise(self, columns: List[str]) -> "DataTransformer":
        self._df = standardise_strings(self._df, columns)
        return self

    @log_execution_time
    def result(self) -> DataFrame:
        """Return the fully transformed DataFrame."""
        logger.info(
            "Transformation complete — schema: %s",
            [f.name for f in self._df.schema.fields],
        )
        return self._df
