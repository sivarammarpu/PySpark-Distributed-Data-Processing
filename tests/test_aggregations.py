"""
tests/test_aggregations.py
===========================
Unit tests for aggregation functions.
"""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F

from src.aggregations.aggregator import (
    DataAggregator,
    apply_window_functions,
    group_aggregate,
    statistical_aggregates,
)


@pytest.fixture()
def agg_df(spark):
    """Small but realistic sales DataFrame for aggregation tests."""
    data = [
        Row(order_id="O1", region="NORTH", product="Laptop",
            customer_id="C1", revenue=1800.0, quantity=2, order_date="2024-01-15"),
        Row(order_id="O2", region="NORTH", product="Monitor",
            customer_id="C2", revenue=500.0,  quantity=1, order_date="2024-01-20"),
        Row(order_id="O3", region="SOUTH", product="Laptop",
            customer_id="C3", revenue=900.0,  quantity=1, order_date="2024-02-10"),
        Row(order_id="O4", region="SOUTH", product="Keyboard",
            customer_id="C1", revenue=160.0,  quantity=4, order_date="2024-02-15"),
        Row(order_id="O5", region="EAST",  product="Mouse",
            customer_id="C4", revenue=75.0,   quantity=3, order_date="2024-03-01"),
    ]
    return spark.createDataFrame(data)


class TestGroupAggregate:

    def test_basic_group_by_region(self, spark, agg_df):
        result = group_aggregate(
            agg_df,
            group_cols=["region"],
            agg_map={"revenue": ["sum", "avg"], "quantity": ["sum"]},
        )
        assert result.count() == 3  # NORTH, SOUTH, EAST
        cols = result.columns
        assert "sum_revenue" in cols
        assert "avg_revenue" in cols
        assert "sum_quantity" in cols

    def test_sum_values_correct(self, spark, agg_df):
        result = group_aggregate(agg_df, ["region"], {"revenue": ["sum"]})
        north = result.filter(F.col("region") == "NORTH") \
                      .select("sum_revenue").first()["sum_revenue"]
        assert abs(north - 2300.0) < 0.01

    def test_multi_column_group(self, spark, agg_df):
        result = group_aggregate(
            agg_df,
            group_cols=["region", "product"],
            agg_map={"revenue": ["max"], "order_id": ["count"]},
        )
        # NORTH has 2 products, SOUTH has 2, EAST has 1 → 5 combinations
        assert result.count() == 5


class TestWindowFunctions:

    def test_window_columns_added(self, spark, agg_df):
        result = apply_window_functions(agg_df, "region", "order_date", "revenue")
        expected = ["row_num", "rank", "dense_rank", "lag_revenue", "lead_revenue", "running_total"]
        for col in expected:
            assert col in result.columns, f"Missing window column: {col}"

    def test_row_number_starts_at_one(self, spark, agg_df):
        result = apply_window_functions(agg_df, "region", "order_date", "revenue")
        min_row = result.agg(F.min("row_num")).first()[0]
        assert min_row == 1

    def test_running_total_monotone_within_partition(self, spark, agg_df):
        result = apply_window_functions(agg_df, "region", "order_date", "revenue")
        north = (
            result
            .filter(F.col("region") == "NORTH")
            .orderBy("order_date")
            .select("running_total")
            .collect()
        )
        totals = [r["running_total"] for r in north]
        assert totals == sorted(totals)


class TestStatisticalAggregates:

    def test_returns_single_row(self, spark, agg_df):
        result = statistical_aggregates(agg_df, ["revenue", "quantity"])
        assert result.count() == 1

    def test_expected_columns_present(self, spark, agg_df):
        result = statistical_aggregates(agg_df, ["revenue"], percentiles=[0.5])
        cols = result.columns
        assert "mean_revenue" in cols
        assert "stddev_revenue" in cols
        assert "min_revenue" in cols
        assert "max_revenue" in cols
        assert "p50_revenue" in cols

    def test_mean_revenue_correct(self, spark, agg_df):
        result = statistical_aggregates(agg_df, ["revenue"])
        mean_val = result.select("mean_revenue").first()[0]
        expected = (1800 + 500 + 900 + 160 + 75) / 5  # 687.0
        assert abs(mean_val - expected) < 1.0

    def test_handles_missing_column(self, spark, agg_df):
        # Should skip missing column gracefully — returned DF has 0 rows
        result = statistical_aggregates(agg_df, ["nonexistent_col_xyz"])
        assert result.count() == 0


class TestDataAggregatorClass:

    def test_builder_group(self, spark, agg_df):
        agg = DataAggregator(agg_df)
        result = agg.group(["region"], {"revenue": ["sum"]})
        assert result.count() == 3

    def test_builder_window(self, spark, agg_df):
        agg = DataAggregator(agg_df)
        result = agg.window("region", "order_date", "revenue")
        assert "row_num" in result.columns

    def test_builder_stats(self, spark, agg_df):
        agg = DataAggregator(agg_df)
        result = agg.stats(["revenue", "quantity"], percentiles=[0.25, 0.75])
        assert result.count() == 1
        assert "p25_revenue" in result.columns
