"""
src/pipeline/pipeline.py
=========================
End-to-end pipeline orchestrator that wires all modules together.

Pipeline phases
---------------
1.  Load raw CSV data into DataFrame
2.  Validate ingested data
3.  Transform (cast, nulls, derive, conditional)
4.  Aggregate (GroupBy, window functions, stats)
5.  Register SparkSQL views & run analytical queries
6.  Optimise partitions + cache hot DataFrames
7.  Write outputs to Parquet with partitioning
8.  Validate written output
9.  Cleanup resources
"""

import time
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession

from src.aggregations.aggregator import DataAggregator
from src.ingestion.data_loader import load_csv, validate_dataframe
from src.optimization.optimizer import DataOptimizer
from src.sql.spark_sql_engine import SparkSQLEngine
from src.storage.parquet_writer import ParquetStorage
from src.transformations.transformer import DataTransformer
from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class DataPipeline:
    """
    Orchestrates the complete data processing pipeline.

    Parameters
    ----------
    spark      : Active SparkSession (injected from ``main.py``).
    input_path : Path to the raw CSV input file.
    output_dir : Root output directory for Parquet files.
    """

    def __init__(
        self,
        spark: SparkSession,
        input_path: str,
        output_dir: str,
    ):
        self.spark       = spark
        self.input_path  = input_path
        self.output_dir  = Path(output_dir)
        self.sql_engine  = SparkSQLEngine(spark)
        self._optimizer  = None

    # ─────────────────────────────────────────────────────────────────────
    @log_execution_time
    def run(self) -> None:
        """Execute all pipeline phases in order."""
        pipeline_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info("  DISTRIBUTED DATA PIPELINE — START")
        logger.info("=" * 70)

        try:
            # ── Phase 1: Ingest ──────────────────────────────────────────
            logger.info("── Phase 1: Data Ingestion ──")
            raw_df = load_csv(
                self.spark,
                path=self.input_path,
                header=True,
                infer_schema=True,
                mode="PERMISSIVE",
            )

            # ── Phase 2: Validate ────────────────────────────────────────
            logger.info("── Phase 2: Validation ──")
            required_cols = [
                "order_id", "order_date", "region", "customer_id",
                "product", "quantity", "unit_price",
            ]
            validate_dataframe(
                raw_df,
                required_columns=required_cols,
                null_threshold=0.5,
                min_row_count=100,
            )

            # ── Phase 3: Transform ───────────────────────────────────────
            logger.info("── Phase 3: Transformations ──")
            transformed_df = (
                DataTransformer(raw_df)
                .filter("quantity > 0 AND unit_price > 0")
                .cast({
                    "quantity":   "integer",
                    "unit_price": "double",
                    "discount":   "double",
                })
                .nulls(strategy="fill", fill_map={"discount": 0.0, "region": "UNKNOWN"})
                .derive()
                .conditional()
                .dedup(["order_id"])
                .standardise(["region", "product", "payment_method"])
                .result()
            )

            # ── Phase 4: Optimise (pre-aggregation) ──────────────────────
            logger.info("── Phase 4: Optimisation ──")
            self._optimizer = DataOptimizer(transformed_df).repartition(50, "region")
            self._optimizer.diagnose(label="post-transform")
            cache_df = self._optimizer.cache("memory_and_disk").result()

            # ── Phase 5: Aggregate ───────────────────────────────────────
            logger.info("── Phase 5: Aggregations ──")
            agg = DataAggregator(cache_df)

            regional_summary = agg.group(
                group_cols=["region", "order_year", "order_month"],
                agg_map={
                    "revenue":  ["sum", "avg", "max"],
                    "quantity": ["sum"],
                    "order_id": ["count"],
                },
            )

            windowed_df = agg.window(
                partition_col="region",
                order_col="order_date",
                value_col="revenue",
            )

            stats_df = agg.stats(
                numeric_cols=["revenue", "quantity", "unit_price"],
                percentiles=[0.25, 0.5, 0.75, 0.95],
            )

            logger.info("Statistical summary:")
            stats_df.show(truncate=False)

            # ── Phase 6: SparkSQL ────────────────────────────────────────
            logger.info("── Phase 6: SparkSQL ──")
            self.sql_engine.register(cache_df,         "sales")
            self.sql_engine.register(regional_summary, "regional_summary")

            top_products_df = self.sql_engine.cte("sales")
            high_value_customers = self.sql_engine.subquery("sales")
            rolling_avg_df = self.sql_engine.analytical("sales")

            logger.info("Top products per region (sample):")
            top_products_df.show(10, truncate=False)

            logger.info("Rolling 3-month average revenue (sample):")
            rolling_avg_df.show(10, truncate=False)

            # ── Phase 7: Write Parquet ───────────────────────────────────
            logger.info("── Phase 7: Parquet Storage ──")

            # Main transformed output
            main_out = str(self.output_dir / "transformed")
            ParquetStorage(self.spark, main_out).write(
                cache_df,
                partition_cols=["order_year", "order_month", "region"],
            )

            # Regional summary
            summary_out = str(self.output_dir / "regional_summary")
            ParquetStorage(self.spark, summary_out).write(
                regional_summary,
                partition_cols=["order_year", "region"],
            )

            # Top products
            top_out = str(self.output_dir / "top_products")
            ParquetStorage(self.spark, top_out).write(top_products_df)

            # Customer analysis
            customers_out = str(self.output_dir / "high_value_customers")
            ParquetStorage(self.spark, customers_out).write(high_value_customers)

            # Window enriched
            window_out = str(self.output_dir / "windowed")
            ParquetStorage(self.spark, window_out).write(windowed_df)

            # ── Phase 8: Validate output ─────────────────────────────────
            logger.info("── Phase 8: Output Validation ──")
            validated = ParquetStorage(self.spark, main_out).validate(
                expected_schema_cols=required_cols + ["revenue", "sales_tier"],
            )
            logger.info("Output row count: %d", validated.count())

        except Exception as exc:
            logger.error("Pipeline FAILED: %s", exc, exc_info=True)
            raise

        finally:
            # ── Phase 9: Cleanup ─────────────────────────────────────────
            logger.info("── Phase 9: Cleanup ──")
            self.sql_engine.drop_all()
            if self._optimizer:
                self._optimizer.cleanup()

            elapsed = time.perf_counter() - pipeline_start
            logger.info("=" * 70)
            logger.info("  PIPELINE COMPLETE  [%.2f s]", elapsed)
            logger.info("=" * 70)
