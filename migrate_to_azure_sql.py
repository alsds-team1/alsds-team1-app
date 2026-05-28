"""
Azure SQL migration script for the Urban AI / Huff Engine project.

What it does:
1. Reads the Worcester CSV files and GeoJSON file.
2. Converts the GeoJSON into a real SQL table: cbg_geojson.
3. Builds normalized tables and precomputed summary tables.
4. Uploads everything to Azure SQL using db.py / SQL_CONNECTION_STRING.

Run locally after setting SQL_CONNECTION_STRING:
    python migrate_to_azure_sql.py
"""

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
import sqlite3

from db import get_connection
import logging

logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data folder
DATA_DIR = BASE_DIR / "Data"


def sanitize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def insert_dataframe(conn, table_name: str, df: pd.DataFrame, columns: List[str], chunk_size: int = 50000) -> None:
    """
    Safely insert DataFrame data into Azure SQL in chunks to prevent memory overflow (OOM).
    :param chunk_size: Number of rows to process and insert per batch. Adjust based on server memory.
    """
    if df.empty:
        return

    placeholders = ", ".join(["?"] * len(columns))
    col_sql = ", ".join(f"[{c}]" for c in columns)
    sql = f"INSERT INTO {table_name} ({col_sql}) VALUES ({placeholders})"
    
    cursor = conn.cursor()
    # Enable fast_executemany to greatly boost bulk insert speed
    cursor.fast_executemany = True 
    
    total_rows = len(df)
    
    # Use range to slice DataFrame by step
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        
        # 1. Slice out the current small chunk of the DataFrame (e.g., 0 to 50000 rows)
        chunk_df = df.iloc[start_idx:end_idx]
        
        # 2. Convert only this small chunk of data to a List; memory footprint remains strictly bounded
        rows = [
            tuple(sanitize_value(v) for v in row) 
            for row in chunk_df[columns].itertuples(index=False, name=None)
        ]
        
        # 3. Insert into database immediately
        if rows:
            try:
                cursor.executemany(sql, rows)
                logger.info("  -> [%s] Successfully inserted rows %d to %d", table_name, start_idx + 1, end_idx)
            except Exception as e:
                logger.exception("  -> [%s] Failed to insert rows %d to %d!", table_name, start_idx + 1, end_idx)
                raise e # Raise exception to trigger outer rollback or error handling
                
    # Once all chunks for this DataFrame are successfully inserted, commit the transaction
    conn.commit()


def execute_statements(conn, statements: Iterable[str]) -> None:
    cursor = conn.cursor()
    for statement in statements:
        try:
            # Log the exact SQL statement being executed (trim long whitespace for readability)
            if isinstance(statement, str):
                stmt_preview = statement.strip()
            else:
                stmt_preview = str(statement)
            logger.info("Executing SQL statement: %s", stmt_preview)
            cursor.execute(statement)
        except Exception:
            # Log full exception with statement context and re-raise for caller to handle
            logger.exception("Error executing SQL statement: %s", stmt_preview)
            raise
    conn.commit()


def load_sqlite_db(sqlite_path: Path) -> dict:
    """Load all non-system tables from a local SQLite database and return a dict of DataFrames."""
    logger.info("Loading local SQLite DB: %s", sqlite_path)
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    cursor = sqlite_conn.cursor()
    # fetch user tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]
    dfs = {}
    for t in tables:
        logger.info("Reading table from sqlite: %s", t)
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', sqlite_conn)
        except Exception:
            logger.exception("Failed to read table %s from sqlite", t)
            raise
        dfs[t] = df
    sqlite_conn.close()
    return dfs


def infer_azure_column_type(series: pd.Series) -> str:
    """Infer a simple Azure SQL column type from a pandas Series."""
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return "INT"
    if pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    if pd.api.types.is_bool_dtype(dtype):
        return "BIT"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    # default to NVARCHAR(MAX) for objects/strings
    return "NVARCHAR(MAX)"


def build_tables() -> Tuple[dict, pd.DataFrame]:
    """Load all tables from Data/team1.db and return dict of DataFrames and a migration summary DataFrame."""
    TEAM1_DB = DATA_DIR / "team1.db"
    if not TEAM1_DB.exists():
        raise FileNotFoundError(f"Local sqlite DB not found: {TEAM1_DB}")
    dfs = load_sqlite_db(TEAM1_DB)

    migration_summary = pd.DataFrame(
        {
            "table_name": list(dfs.keys()),
            "row_count": [len(df) for df in dfs.values()],
        }
    )

    return dfs, migration_summary


def migrate() -> dict:
    try:
        dfs, migration_summary = build_tables()

        logger.info("Connecting to Azure SQL...")
        with get_connection() as conn:
            # Drop existing tables (we'll drop only the ones present in the sqlite db)
            # use schema-qualified names and bracket-identifiers to be safe
            drop_statements = [f"DROP TABLE IF EXISTS [dbo].[{t}]" for t in dfs.keys()] + ["DROP TABLE IF EXISTS [dbo].[migration_summary]"]
            logger.info("Dropping old tables if they exist: %s", list(dfs.keys()))
            execute_statements(conn, drop_statements)

            # Create tables inferred from DataFrame dtypes
            create_statements = []
            for table_name, df in dfs.items():
                cols = []
                for col in df.columns:
                    col_type = infer_azure_column_type(df[col])
                    # allow NULLs for everything by default; primary keys are not inferred automatically
                    cols.append(f"[{col}] {col_type} NULL")
                # create table only if it does not exist
                create_sql = (
                    f"IF OBJECT_ID(N'[dbo].[{table_name}]', N'U') IS NULL "
                    f"BEGIN CREATE TABLE [dbo].[{table_name}] (" + ", ".join(cols) + ") END"
                )
                create_statements.append(create_sql)

            # Always create migration_summary table
            create_statements.append(
                "IF OBJECT_ID(N'[dbo].[migration_summary]', N'U') IS NULL BEGIN CREATE TABLE [dbo].[migration_summary] (table_name NVARCHAR(100) NOT NULL, row_count INT NOT NULL) END"
            )

            logger.info("Creating tables on Azure SQL: %s", list(dfs.keys()))
            execute_statements(conn, create_statements)

            logger.info("Uploading data to Azure SQL...")
            for table_name, df in dfs.items():
                if df.empty:
                    logger.info("Skipping empty table: %s", table_name)
                    continue
                logger.info("Inserting %d rows into %s", len(df), table_name)
                insert_dataframe(conn, f"[dbo].[{table_name}]", df, list(df.columns))

            # Insert migration summary into schema-qualified table
            insert_dataframe(conn, "[dbo].[migration_summary]", migration_summary, list(migration_summary.columns))

        logger.info("\nSUCCESS: Azure SQL migration completed.")
        logger.info('\n' + migration_summary.to_string(index=False))
        return {"ok": True, "migration_summary": migration_summary.to_dict(orient="records")}
    except Exception as e:
        logger.exception("Migration failed: %s", e)
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # run migration when invoked directly; print the result for convenience
    result = migrate()
    print(result)
