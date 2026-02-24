# System Architecture — Distributed Data Processing Pipeline

## Overview

This system is a modular, layered PySpark pipeline with nine execution phases. Each layer has a single responsibility and can be tested, replaced, or scaled independently.

---

## Architecture Diagram

```mermaid
graph TD
    subgraph Sources["Data Sources"]
        CSV["CSV Files"]
        JSON["JSON Files"]
        JDBC["JDBC / RDBMS"]
        Cloud["Cloud Storage\nS3 / Azure / GCS"]
    end

    subgraph Ingestion["Phase 1–2: Ingest & Validate"]
        DL["data_loader.py\n• load_csv / load_json / load_jdbc / load_cloud\n• validate_dataframe\n• load_incremental"]
    end

    subgraph Transform["Phase 3: Transform"]
        TR["transformer.py\n• filter_records\n• cast_columns / handle_nulls\n• add_derived_columns\n• apply_conditional\n• DataTransformer builder"]
    end

    subgraph Optimise["Phase 4: Optimise"]
        OPT["optimizer.py\n• smart_repartition / coalesce\n• cache_dataframe\n• log_partition_info"]
    end

    subgraph Aggregate["Phase 5: Aggregate"]
        AGG["aggregator.py\n• group_aggregate\n• apply_window_functions\n• statistical_aggregates"]
    end

    subgraph SQL["Phase 6: SparkSQL"]
        SQL_E["spark_sql_engine.py\n• register_view\n• CTE / subquery / analytical\n• complex JOIN queries"]
    end

    subgraph Storage["Phase 7–8: Write & Validate"]
        PW["parquet_writer.py\n• write_parquet — Snappy + partition\n• read_and_validate_parquet"]
    end

    subgraph Output["Output Parquet Store"]
        P1["transformed/\n  order_year / order_month / region"]
        P2["regional_summary/"]
        P3["top_products/"]
        P4["high_value_customers/"]
        P5["windowed/"]
    end

    subgraph Support["Cross-cutting Concerns"]
        LOG["logger.py\nLog4j-style + timing decorator"]
        RET["retry.py\nExponential back-off"]
        CFG["spark_config.py\nSparkSession + AQE + Kryo"]
    end

    CSV & JSON & JDBC & Cloud --> DL
    DL --> TR
    TR --> OPT
    OPT --> AGG
    AGG --> SQL_E
    SQL_E --> PW
    PW --> P1 & P2 & P3 & P4 & P5
    LOG & RET & CFG -.-> DL & TR & OPT & AGG & SQL_E & PW
```

---

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant U as User / CI-CD
    participant M as main.py
    participant C as spark_config.py
    participant P as pipeline.py
    participant DL as data_loader
    participant T as transformer
    participant O as optimizer
    participant A as aggregator
    participant S as spark_sql_engine
    participant W as parquet_writer

    U->>M: python main.py --input ... --output ... --env local
    M->>C: build_spark_session(env="local")
    C-->>M: SparkSession
    M->>P: DataPipeline.run()

    P->>DL: load_csv(path, mode=PERMISSIVE)
    DL-->>P: raw_df

    P->>DL: validate_dataframe(raw_df, required_cols)
    DL-->>P: validated_df ✔

    P->>T: DataTransformer(df).filter().cast().nulls().derive().conditional().dedup()
    T-->>P: transformed_df

    P->>O: DataOptimizer(df).repartition(50,"region").cache().diagnose()
    O-->>P: cached_df

    P->>A: group_aggregate / window_functions / statistical_aggregates
    A-->>P: summary_dfs

    P->>S: register_view + cte / subquery / analytical
    S-->>P: result_dfs

    P->>W: write_parquet(df, partition_cols=[year,month,region])
    W-->>P: Parquet on disk ✔

    P->>W: read_and_validate_parquet(path, expected_schema_cols)
    W-->>P: validated ✔

    P->>M: Pipeline complete
    M->>C: stop_spark_session()
```

---

## Module Responsibility Matrix

| Module | Reads | Writes | External I/O |
|---|---|---|---|
| `spark_config.py` | — | SparkSession config | None |
| `data_loader.py` | Raw files (CSV/JSON/JDBC/Cloud) | DataFrame | File system / JDBC / Object store |
| `transformer.py` | DataFrame | DataFrame | None |
| `aggregator.py` | DataFrame | DataFrame | None |
| `spark_sql_engine.py` | DataFrame | DataFrame | Spark Catalog |
| `optimizer.py` | DataFrame | DataFrame | None |
| `parquet_writer.py` | DataFrame | Parquet files | File system / Object store |
| `pipeline.py` | All modules | Pipeline result | Logs |
| `logger.py` | — | Log statements | stdout |
| `retry.py` | — | — | None |

---

## Partitioning Strategy

```
data/output/transformed/
  └── order_year=2022/
        └── order_month=1/
              └── region=NORTH/
                    └── part-00000.snappy.parquet
```

### Why this layout?

| Benefit | Mechanism |
|---|---|
| **Partition pruning** | Queries filtering on `order_year`, `order_month`, or `region` skip irrelevant directories entirely |
| **Predicate pushdown** | Parquet's row-group statistics allow skipping blocks within a file |
| **Dynamic overwrite** | Only partitions present in the new data are replaced — historical partitions untouched |
| **Columnar compression** | Snappy gives ~2–3× compression ratio with low CPU overhead |

---

## Cluster Deployment Modes

| Mode | Master URL | Use Case |
|---|---|---|
| Local (dev) | `local[*]` | Development on a single machine |
| Standalone | `spark://host:7077` | Small dedicated Spark cluster |
| YARN | `yarn` | Hadoop ecosystem (EMR, HDInsight) |
| Kubernetes | `k8s://https://...` | Cloud-native container orchestration |

All modes are parameterised via the `--env` flag in `main.py` and the `MASTER` env-var in `deploy/submit_job.sh`.
