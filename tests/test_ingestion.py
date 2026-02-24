"""
tests/test_ingestion.py
========================
Unit tests for data ingestion functions.
"""

import os
import tempfile

import pytest
from pyspark.sql import Row
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from src.ingestion.data_loader import (
    load_csv,
    load_incremental,
    load_json,
    validate_dataframe,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def make_csv(content: str) -> str:
    """Write content to a temp CSV file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return f.name


def make_json(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return f.name


# ── CSV Tests ──────────────────────────────────────────────────────────────

class TestLoadCSV:

    def test_basic_csv_load(self, spark):
        path = make_csv("id,name,value\n1,Alice,100\n2,Bob,200\n")
        df = load_csv(spark, path, infer_schema=True)
        assert df.count() == 2
        assert set(df.columns) == {"id", "name", "value"}
        os.unlink(path)

    def test_manual_schema(self, spark):
        schema = StructType([
            StructField("id",    IntegerType(), True),
            StructField("name",  StringType(),  True),
            StructField("value", DoubleType(),  True),
        ])
        path = make_csv("id,name,value\n1,Alice,100.5\n2,Bob,200.0\n")
        df = load_csv(spark, path, schema=schema)
        assert dict(df.dtypes)["value"] == "double"
        os.unlink(path)

    def test_no_header(self, spark):
        path = make_csv("1,Alice,100\n2,Bob,200\n")
        df = load_csv(spark, path, header=False, infer_schema=False)
        assert df.count() == 2
        os.unlink(path)

    def test_permissive_mode_corrupt_record(self, spark):
        path = make_csv("id,name,value\n1,Alice,100\nBAD_ROW_NO_COLS\n3,Carol,300\n")
        df = load_csv(spark, path, mode="PERMISSIVE")
        # PERMISSIVE keeps all rows (corrupt row stored in _corrupt_record)
        assert df.count() >= 2
        os.unlink(path)


# ── JSON Tests ─────────────────────────────────────────────────────────────

class TestLoadJSON:

    def test_basic_json_load(self, spark):
        path = make_json(
            '{"id": 1, "event": "click"}\n{"id": 2, "event": "view"}\n'
        )
        df = load_json(spark, path)
        assert df.count() == 2
        assert "event" in df.columns
        os.unlink(path)


# ── Validation Tests ───────────────────────────────────────────────────────

class TestValidateDataframe:

    def _create_df(self, spark, data, schema):
        return spark.createDataFrame(data, schema)

    def test_passes_valid_dataframe(self, spark):
        df = spark.createDataFrame(
            [Row(id=1, name="Alice"), Row(id=2, name="Bob")]
        )
        result = validate_dataframe(df, required_columns=["id", "name"])
        assert result.count() == 2

    def test_raises_on_missing_column(self, spark):
        df = spark.createDataFrame([Row(id=1)])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, required_columns=["id", "missing_col"])

    def test_raises_on_too_few_rows(self, spark):
        df = spark.createDataFrame([Row(id=1)])
        with pytest.raises(ValueError, match="minimum required"):
            validate_dataframe(df, required_columns=["id"], min_row_count=10)

    def test_raises_on_high_null_ratio(self, spark):
        df = spark.createDataFrame(
            [Row(id=None), Row(id=None), Row(id=None), Row(id=1)]
        )
        with pytest.raises(ValueError, match="nulls"):
            validate_dataframe(df, required_columns=["id"], null_threshold=0.5)


# ── Incremental Load Tests ─────────────────────────────────────────────────

class TestLoadIncremental:

    def test_incremental_filter(self, spark, tmp_path):
        data = [
            Row(id=1, updated_at="2023-01-01"),
            Row(id=2, updated_at="2023-06-01"),
            Row(id=3, updated_at="2024-01-01"),
        ]
        df = spark.createDataFrame(data)
        parquet_path = str(tmp_path / "test_incremental")
        df.write.parquet(parquet_path)

        result = load_incremental(
            spark,
            path=parquet_path,
            watermark_column="updated_at",
            last_processed_value="2023-03-01",
            file_format="parquet",
        )
        assert result.count() == 2  # id=2 and id=3
