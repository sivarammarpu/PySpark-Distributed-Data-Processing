"""
src/sql/spark_sql_engine.py
============================
SparkSQL engine — registers views and executes structured SQL queries.

Supports:
  - Temporary and global temporary views
  - Complex joins (inner, left, right, full outer)
  - CTEs (WITH clauses)
  - Subqueries
  - Analytical window SQL
"""

from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# View Registration
# ═══════════════════════════════════════════════════════════════════════════

def register_view(
    df: DataFrame,
    view_name: str,
    global_view: bool = False,
) -> None:
    """
    Register a DataFrame as a Spark SQL view.

    Parameters
    ----------
    df          : DataFrame to expose as a SQL view.
    view_name   : SQL-accessible name.
    global_view : If True, creates a ``global_temp.{view_name}`` that
                  persists across sessions; otherwise a session-scoped temp view.
    """
    if global_view:
        df.createOrReplaceGlobalTempView(view_name)
        logger.info("Registered global temp view: global_temp.%s", view_name)
    else:
        df.createOrReplaceTempView(view_name)
        logger.info("Registered temp view: %s", view_name)


def drop_view(spark: SparkSession, view_name: str, global_view: bool = False) -> None:
    """Drop a registered view."""
    try:
        if global_view:
            spark.catalog.dropGlobalTempView(view_name)
            logger.info("Dropped global temp view: global_temp.%s", view_name)
        else:
            spark.catalog.dropTempView(view_name)
            logger.info("Dropped temp view: %s", view_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not drop view '%s': %s", view_name, exc)


# ═══════════════════════════════════════════════════════════════════════════
# Join Queries
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def run_join_query(
    spark: SparkSession,
    left_view: str,
    right_view: str,
    join_col: str,
    join_type: str = "inner",
    select_cols: str = "*",
    filter_clause: str = "",
) -> DataFrame:
    """
    Execute a configurable JOIN between two registered views.

    Parameters
    ----------
    spark         : Active SparkSession.
    left_view     : Name of the left view.
    right_view    : Name of the right view.
    join_col      : Column name used for the join predicate (must exist in both).
    join_type     : ``inner``, ``left``, ``right``, or ``full``.
    select_cols   : SQL SELECT expression (default ``*``).
    filter_clause : Optional WHERE clause (without the ``WHERE`` keyword).

    Returns
    -------
    DataFrame
    """
    join_keyword = {
        "inner": "INNER JOIN",
        "left":  "LEFT JOIN",
        "right": "RIGHT JOIN",
        "full":  "FULL OUTER JOIN",
    }.get(join_type.lower(), "INNER JOIN")

    where_clause = f"WHERE {filter_clause}" if filter_clause else ""

    sql = f"""
        SELECT {select_cols}
        FROM   {left_view}  l
        {join_keyword}
               {right_view} r
          ON l.{join_col} = r.{join_col}
        {where_clause}
    """
    logger.info("Executing JOIN query [%s] …\n%s", join_type.upper(), sql.strip())
    result = spark.sql(sql)
    logger.info("JOIN query returned %d rows", result.count())
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CTE Query
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def run_cte_query(spark: SparkSession, sales_view: str) -> DataFrame:
    """
    Execute a multi-CTE analytical query on the *sales_view*.

    CTEs used
    ---------
    1. ``regional_totals``  — total revenue and orders per region.
    2. ``ranked_products``  — products ranked by revenue within each region.

    Returns
    -------
    DataFrame  — top 5 products per region by revenue.
    """
    sql = f"""
        WITH regional_totals AS (
            SELECT
                region,
                SUM(revenue)   AS total_revenue,
                COUNT(order_id) AS total_orders
            FROM {sales_view}
            GROUP BY region
        ),
        ranked_products AS (
            SELECT
                s.region,
                s.product,
                SUM(s.revenue)          AS product_revenue,
                COUNT(s.order_id)       AS product_orders,
                rt.total_revenue,
                ROUND(
                    SUM(s.revenue) / NULLIF(rt.total_revenue, 0) * 100, 2
                ) AS revenue_share_pct,
                DENSE_RANK() OVER (
                    PARTITION BY s.region
                    ORDER BY SUM(s.revenue) DESC
                ) AS product_rank
            FROM {sales_view} s
            JOIN regional_totals rt ON s.region = rt.region
            GROUP BY s.region, s.product, rt.total_revenue
        )
        SELECT *
        FROM   ranked_products
        WHERE  product_rank <= 5
        ORDER  BY region, product_rank
    """
    logger.info("Executing CTE query on view: %s", sales_view)
    result = spark.sql(sql)
    logger.info("CTE query complete — %d rows", result.count())
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Subquery
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def run_subquery(spark: SparkSession, sales_view: str) -> DataFrame:
    """
    Find customers whose total spend exceeds the overall average order value
    using a subquery.

    Returns
    -------
    DataFrame  — customer_id, total_spend, avg_order_value_overall
    """
    sql = f"""
        SELECT
            customer_id,
            ROUND(SUM(revenue), 2)  AS total_spend,
            COUNT(order_id)         AS num_orders,
            ROUND(
                (SELECT AVG(revenue) FROM {sales_view}), 2
            ) AS avg_order_overall
        FROM {sales_view}
        GROUP BY customer_id
        HAVING SUM(revenue) > (SELECT AVG(revenue) FROM {sales_view})
        ORDER BY total_spend DESC
    """
    logger.info("Executing subquery on view: %s", sales_view)
    result = spark.sql(sql)
    logger.info("Subquery complete — %d high-value customers", result.count())
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Analytical Window SQL
# ═══════════════════════════════════════════════════════════════════════════

@log_execution_time
def run_analytical_query(spark: SparkSession, sales_view: str) -> DataFrame:
    """
    Window-function-based analytical query:
    computes a 3-month rolling average revenue per region.

    Returns
    -------
    DataFrame  — region, order_year, order_month, monthly_revenue,
                 rolling_3m_avg_revenue
    """
    sql = f"""
        WITH monthly AS (
            SELECT
                region,
                order_year,
                order_month,
                ROUND(SUM(revenue), 2) AS monthly_revenue
            FROM {sales_view}
            GROUP BY region, order_year, order_month
        )
        SELECT
            region,
            order_year,
            order_month,
            monthly_revenue,
            ROUND(
                AVG(monthly_revenue) OVER (
                    PARTITION BY region
                    ORDER BY order_year, order_month
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ), 2
            ) AS rolling_3m_avg_revenue
        FROM monthly
        ORDER BY region, order_year, order_month
    """
    logger.info("Executing analytical (rolling avg) query on view: %s", sales_view)
    result = spark.sql(sql)
    logger.info("Analytical query complete — %d rows", result.count())
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SparkSQLEngine — orchestration class
# ═══════════════════════════════════════════════════════════════════════════

class SparkSQLEngine:
    """
    Manages view registration, query execution, and view lifecycle.

    Usage
    -----
    >>> engine = SparkSQLEngine(spark)
    >>> engine.register(df, "sales")
    >>> result = engine.cte("sales")
    >>> engine.drop("sales")
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._registered_views: list[str] = []

    def register(self, df: DataFrame, view_name: str, global_view: bool = False) -> None:
        register_view(df, view_name, global_view)
        self._registered_views.append(view_name)

    def join(self, left: str, right: str, on: str,
             how: str = "inner", select: str = "*", where: str = "") -> DataFrame:
        return run_join_query(self.spark, left, right, on, how, select, where)

    def cte(self, sales_view: str) -> DataFrame:
        return run_cte_query(self.spark, sales_view)

    def subquery(self, sales_view: str) -> DataFrame:
        return run_subquery(self.spark, sales_view)

    def analytical(self, sales_view: str) -> DataFrame:
        return run_analytical_query(self.spark, sales_view)

    def sql(self, query: str) -> DataFrame:
        """Execute an arbitrary SQL string."""
        logger.info("Executing raw SQL …")
        return self.spark.sql(query)

    def drop(self, view_name: str, global_view: bool = False) -> None:
        drop_view(self.spark, view_name, global_view)

    def drop_all(self) -> None:
        """Drop all views registered by this engine instance."""
        for v in self._registered_views:
            try:
                drop_view(self.spark, v)
            except Exception:  # noqa: BLE001
                pass
        self._registered_views.clear()
