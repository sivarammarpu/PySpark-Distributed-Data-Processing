"""
tests/test_transformations.py
==============================
Unit tests for transformer functions.
"""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F

from src.transformations.transformer import (
    DataTransformer,
    add_derived_columns,
    apply_conditional,
    cast_columns,
    drop_duplicate_records,
    filter_records,
    handle_nulls,
    select_rename,
    standardise_strings,
)


@pytest.fixture()
def sales_df(spark):
    data = [
        Row(order_id="O1", order_date="2024-03-15", region="north",
            customer_id="C1", product="laptop", quantity=2,
            unit_price=999.99, discount=0.10, payment_method="Credit Card"),
        Row(order_id="O2", order_date="2024-06-20", region="south",
            customer_id="C2", product="monitor", quantity=1,
            unit_price=499.99, discount=0.0, payment_method="Cash"),
        Row(order_id="O3", order_date="2024-09-01", region="east",
            customer_id="C3", product="keyboard", quantity=5,
            unit_price=79.99, discount=0.05, payment_method="Digital Wallet"),
        # duplicate
        Row(order_id="O1", order_date="2024-03-15", region="north",
            customer_id="C1", product="laptop", quantity=2,
            unit_price=999.99, discount=0.10, payment_method="Credit Card"),
    ]
    return spark.createDataFrame(data)


class TestFilterRecords:

    def test_quantity_filter(self, spark, sales_df):
        result = filter_records(sales_df, "quantity > 1")
        assert result.count() == 3  # O1(x2 including dup), O3

    def test_returns_empty_when_no_match(self, spark, sales_df):
        result = filter_records(sales_df, "quantity > 100")
        assert result.count() == 0


class TestSelectRename:

    def test_rename_columns(self, spark, sales_df):
        result = select_rename(sales_df, {"order_id": "id", "region": "zone"})
        assert "id" in result.columns
        assert "zone" in result.columns
        assert "order_id" not in result.columns

    def test_column_count(self, spark, sales_df):
        result = select_rename(sales_df, {"order_id": "id", "quantity": "qty"})
        assert len(result.columns) == 2


class TestCastColumns:

    def test_cast_to_double(self, spark, sales_df):
        result = cast_columns(sales_df, {"unit_price": "string"})
        assert dict(result.dtypes)["unit_price"] == "string"

    def test_cast_multiple(self, spark, sales_df):
        result = cast_columns(sales_df, {"quantity": "long", "discount": "float"})
        dtypes = dict(result.dtypes)
        assert dtypes["quantity"] == "bigint"
        assert dtypes["discount"] == "float"


class TestHandleNulls:

    def test_fill_strategy(self, spark):
        df = spark.createDataFrame([Row(a=None, b=5), Row(a=3, b=None)])
        result = handle_nulls(df, strategy="fill", fill_map={"a": 0, "b": 0})
        nulls_a = result.filter(F.col("a").isNull()).count()
        nulls_b = result.filter(F.col("b").isNull()).count()
        assert nulls_a == 0
        assert nulls_b == 0

    def test_drop_strategy(self, spark):
        df = spark.createDataFrame([Row(a=None, b=5), Row(a=3, b=4)])
        result = handle_nulls(df, strategy="drop", subset=["a"])
        assert result.count() == 1


class TestDerivedColumns:

    def test_revenue_computed(self, spark, sales_df):
        result = add_derived_columns(sales_df)
        assert "revenue" in result.columns
        assert "ingestion_ts" in result.columns

    def test_date_parts_extracted(self, spark, sales_df):
        result = add_derived_columns(sales_df)
        assert "order_year" in result.columns
        assert "order_month" in result.columns
        assert "order_quarter" in result.columns

    def test_revenue_value_correct(self, spark, sales_df):
        result = add_derived_columns(sales_df)
        row = result.filter(F.col("order_id") == "O2").select("revenue").first()
        # O2: qty=1, price=499.99, discount=0 → revenue=499.99
        assert abs(row["revenue"] - 499.99) < 0.01


class TestConditionalTransformations:

    def test_sales_tier_column(self, spark, sales_df):
        df = add_derived_columns(sales_df)
        result = apply_conditional(df)
        assert "sales_tier" in result.columns
        tiers = [r["sales_tier"] for r in result.select("sales_tier").collect()]
        assert all(t in ("Bronze", "Silver", "Gold", "Platinum") for t in tiers)

    def test_payment_flag(self, spark, sales_df):
        df = add_derived_columns(sales_df)
        result = apply_conditional(df)
        assert "payment_flag" in result.columns
        # Credit Card and Digital Wallet → flag=1
        cc_flag = result.filter(F.col("payment_method") == "Credit Card") \
                        .select("payment_flag").first()["payment_flag"]
        assert cc_flag == 1


class TestDedup:

    def test_removes_duplicates(self, spark, sales_df):
        result = drop_duplicate_records(sales_df, subset=["order_id"])
        assert result.count() == 3  # O1 duplicate removed


class TestStandardiseStrings:

    def test_upper_and_trim(self, spark, sales_df):
        result = standardise_strings(sales_df, ["region", "product"])
        regions = [r["region"] for r in result.select("region").distinct().collect()]
        assert all(r == r.upper() for r in regions)


class TestDataTransformerBuilder:

    def test_full_chain(self, spark, sales_df):
        result = (
            DataTransformer(sales_df)
            .filter("quantity > 0")
            .cast({"quantity": "integer", "unit_price": "double"})
            .nulls(strategy="fill", fill_map={"discount": 0.0})
            .derive()
            .conditional()
            .dedup(["order_id"])
            .standardise(["region"])
            .result()
        )
        assert result.count() == 3
        assert "revenue" in result.columns
        assert "sales_tier" in result.columns
