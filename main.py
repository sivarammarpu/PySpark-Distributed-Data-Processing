"""
main.py
=======
CLI entry point for the Distributed Data Pipeline.

Usage
-----
    # Development (local mode, default paths)
    python main.py

    # Custom paths and environment
    python main.py --input data/raw/sales_data.csv --output data/output/ --env local

    # Staging / Production (master set via spark-submit --master flag)
    spark-submit main.py --input s3a://bucket/raw/ --output s3a://bucket/processed/ --env prod
"""

import argparse
import sys

from config.spark_config import build_spark_session, stop_spark_session
from src.pipeline.pipeline import DataPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed Data Processing Pipeline — PySpark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/raw/sales_data.csv",
        help="Path to input CSV file or directory (local or cloud URI)",
    )
    parser.add_argument(
        "--output",
        default="data/output/",
        help="Root output directory for Parquet files",
    )
    parser.add_argument(
        "--env",
        choices=["local", "staging", "prod"],
        default="local",
        help="Execution environment",
    )
    parser.add_argument(
        "--app-name",
        default="DistributedDataPipeline",
        help="Spark application name (visible in Spark UI)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Starting pipeline — env: %s  input: %s  output: %s",
                args.env, args.input, args.output)

    spark = None
    exit_code = 0

    try:
        spark = build_spark_session(
            app_name=args.app_name,
            env=args.env,
        )

        pipeline = DataPipeline(
            spark=spark,
            input_path=args.input,
            output_dir=args.output,
        )
        pipeline.run()

    except Exception as exc:
        logger.error("Fatal pipeline error: %s", exc, exc_info=True)
        exit_code = 1

    finally:
        if spark:
            stop_spark_session(spark)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
