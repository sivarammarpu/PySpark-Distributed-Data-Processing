"""
tests/conftest.py
==================
Shared PyTest fixtures — SparkSession for unit tests.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession shared across all tests in the session."""
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("PipelineTests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
