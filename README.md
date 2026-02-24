# Distributed Data Processing System Using PySpark

<div align="center">

![PySpark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Parquet](https://img.shields.io/badge/Storage-Parquet-50ABF1?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![CI](https://img.shields.io/github/actions/workflow/status/sivarammarpu/PySpark-Distributed-Data-Processing/ci.yml?style=for-the-badge&label=CI)

**Designed and developed by [Sivaram Marpu](https://github.com/sivarammarpu)**

</div>

---

## 👤 Project Ownership

| Field | Details |
|---|---|
| **Author** | Sivaram Marpu |
| **GitHub** | [@sivarammarpu](https://github.com/sivarammarpu) |
| **Repository** | [PySpark-Distributed-Data-Processing](https://github.com/sivarammarpu/PySpark-Distributed-Data-Processing) |
| **License** | MIT — © 2024 Sivaram Marpu. All rights reserved. |
| **Project Type** | Enterprise-Grade Big Data Engineering |

> This project is the sole intellectual property of **Sivaram Marpu**.
> Unauthorised use, copying, or distribution without explicit permission is prohibited.

---

## Overview

An enterprise-grade, scalable, fault-tolerant distributed data processing pipeline built with **Apache Spark 3.5 (PySpark)**. Ingests CSV/JSON data, performs distributed transformations and aggregations via the DataFrame API and SparkSQL, applies execution-plan optimisations, and writes Parquet outputs with intelligent partitioning.

---

## Quick Start

```bash
# 1. Install dependencies (requires Python 3.9+ and Java 11+)
pip install -r requirements.txt

# 2. Generate sample data (50 000 sales rows + 10 000 event rows)
python data/generate_sample_data.py

# 3. Run the full pipeline (local mode)
python main.py

# 4. Run unit tests
pytest tests/ -v --tb=short
```

Output Parquet files land in `data/output/` partitioned by `order_year / order_month / region`.

---

## Project Structure

```
PySpark Distributed Data Processing/
├── config/
│   └── spark_config.py          # SparkSession builder (env-aware, Kryo, AQE)
├── src/
│   ├── ingestion/
│   │   └── data_loader.py       # CSV, JSON, JDBC, Cloud, incremental loading
│   ├── transformations/
│   │   └── transformer.py       # Filter, cast, nulls, derived cols, conditional
│   ├── aggregations/
│   │   └── aggregator.py        # GroupBy, window funcs, statistics
│   ├── sql/
│   │   └── spark_sql_engine.py  # Temp/global views, CTEs, joins, subqueries
│   ├── optimization/
│   │   └── optimizer.py         # Repartition, broadcast, caching, salt join
│   ├── storage/
│   │   └── parquet_writer.py    # Snappy Parquet, dynamic partitioning, validation
│   ├── pipeline/
│   │   └── pipeline.py          # End-to-end orchestrator (9 phases)
│   └── utils/
│       ├── logger.py            # Log4j-style structured logger + timing decorator
│       └── retry.py             # Exponential-backoff retry decorator
├── data/
│   ├── generate_sample_data.py  # Realistic CSV + JSON data generator
│   └── raw/                     # (created at runtime)
├── tests/
│   ├── conftest.py              # Shared local SparkSession fixture
│   ├── test_ingestion.py        # CSV/JSON load, validation, incremental
│   ├── test_transformations.py  # All transformer functions + builder chain
│   └── test_aggregations.py    # GroupBy, window, stats, DataAggregator
├── deploy/
│   ├── submit_job.sh            # spark-submit for YARN / Kubernetes
│   └── docker/Dockerfile        # Python 3.11 + Java 17 container image
├── .github/workflows/ci.yml     # GitHub Actions — test on every push/PR
├── docs/architecture.md         # System architecture & data flow
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Configuration

All Spark tuning is centralised in `config/spark_config.py`.

| Setting | Value | Rationale |
|---|---|---|
| `spark.executor.memory` | 4 g | Comfortable worker heap |
| `spark.executor.cores` | 2 | Balanced CPU/memory ratio |
| `spark.sql.shuffle.partitions` | 200 (8 in local) | Right-sized shuffle |
| `spark.serializer` | KryoSerializer | ~5× faster than Java serialization |
| `spark.sql.adaptive.enabled` | true | AQE auto-tunes at runtime |
| `spark.sql.adaptive.coalescePartitions` | true | Merges small post-shuffle partitions |
| `spark.sql.sources.partitionOverwriteMode` | dynamic | Safe partition-level overwrite |

Override at runtime:

```bash
python main.py --env prod
```

---

## Pipeline Phases

| # | Phase | Module |
|---|---|---|
| 1 | **Ingest** | `data_loader.py` — CSV/JSON/JDBC/Cloud, corrupt-record handling |
| 2 | **Validate** | `data_loader.validate_dataframe` — column presence, null %, row count |
| 3 | **Transform** | `transformer.py` — filter, cast, nulls, derived cols, conditional tiers |
| 4 | **Optimise** | `optimizer.py` — repartition by region, cache to `MEMORY_AND_DISK` |
| 5 | **Aggregate** | `aggregator.py` — GroupBy, window functions, statistical summaries |
| 6 | **SparkSQL** | `spark_sql_engine.py` — CTE ranking, subquery, rolling 3-month avg |
| 7 | **Write** | `parquet_writer.py` — Snappy Parquet, partitioned by year/month/region |
| 8 | **Validate** | `parquet_writer.read_and_validate_parquet` — schema + row-count check |
| 9 | **Cleanup** | Unpersist cache, drop SQL views |

---

## Performance Optimisation Techniques

### Repartition vs Coalesce

`optimizer.smart_repartition` automatically chooses:

- **Coalesce** when reducing partition count (no shuffle)
- **Repartition** when increasing (full shuffle, better parallelism)

### Broadcast Join

```python
from src.optimization.optimizer import broadcast_join
result = broadcast_join(df_large, df_dimension, "product_id")
```

### Caching

```python
from src.optimization.optimizer import cache_dataframe, unpersist_dataframe
hot_df = cache_dataframe(df, storage_level="memory_and_disk")
unpersist_dataframe(hot_df)
```

### Salt Join (Skew Mitigation)

```python
from src.optimization.optimizer import salt_join
result = salt_join(df_large, df_small, join_col="customer_id", salt_buckets=10)
```

---

## Data Sources

| Format | Loader | Key Options |
|---|---|---|
| CSV | `load_csv()` | schema, header, delimiter, mode |
| JSON | `load_json()` | multiline, mode |
| JDBC | `load_jdbc()` | url, table/query, partition column |
| Cloud | `load_cloud()` | S3 (`s3a://`), Azure (`wasbs://`), GCS (`gs://`) |

---

## Testing

```bash
# All tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Individual suites
pytest tests/test_ingestion.py -v       # 8 tests
pytest tests/test_transformations.py -v # 15 tests
pytest tests/test_aggregations.py -v    # 12 tests
```

Tests run on a local `SparkSession` (`local[2]`) — no cluster required.

---

## Deployment

### Local

```bash
python main.py --env local
```

### YARN

```bash
export MASTER=yarn INPUT=s3a://bucket/raw/ OUTPUT=s3a://bucket/processed/
bash deploy/submit_job.sh
```

### Docker

```bash
docker build -t pyspark-pipeline -f deploy/docker/Dockerfile .
docker run --rm pyspark-pipeline
```

---

## Storage & Partitioning

```
data/output/transformed/
  order_year=2024/
    order_month=1/
      region=NORTH/
        part-00000.snappy.parquet
```

- **Partition pruning** — filters on year/month/region skip entire directories
- **Predicate pushdown** — Parquet's columnar format enables filter-before-read
- **Dynamic overwrite** — only the partitions in the new data are replaced

---

## Future Enhancements

- [ ] Kafka Streaming integration
- [ ] Delta Lake for ACID compliance
- [ ] Spark MLlib model integration
- [ ] Automated data quality monitoring (Great Expectations)
- [ ] Power BI / Tableau dashboard connector

---

## License & Copyright

```
MIT License

Copyright (c) 2024 Sivaram Marpu
GitHub: https://github.com/sivarammarpu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">
  <strong>Built with ❤️ by <a href="https://github.com/sivarammarpu">Sivaram Marpu</a></strong>
</div>
