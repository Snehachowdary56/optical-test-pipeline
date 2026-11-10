from pathlib import Path

import duckdb
import pandas as pd


def save_to_duckdb(db_path: Path, measurements: pd.DataFrame, device_summary: pd.DataFrame) -> None:
    """Stores cleaned measurements and the per-device summary as SQL tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.register("measurements_df", measurements)
        con.register("summary_df", device_summary)
        con.execute("CREATE OR REPLACE TABLE measurements AS SELECT * FROM measurements_df")
        con.execute("CREATE OR REPLACE TABLE device_summary AS SELECT * FROM summary_df")


def query(db_path: Path, sql: str) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(sql).df()


# Example SQL an engineer might run against the database.
FAILING_DEVICES_BY_STATION = """
    SELECT station,
           COUNT(*)                                   AS devices,
           CAST(SUM(CASE WHEN passed THEN 0 ELSE 1 END) AS INTEGER) AS failed,
           ROUND(AVG(max_loss_db), 3)                 AS avg_worst_loss_db
    FROM device_summary
    GROUP BY station
    ORDER BY station
"""
