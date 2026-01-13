"""Utility functions for interacting with the SQLite database."""
import io
import logging
import os
import sqlite3
from typing import Any, List, Optional

import torch


def select_tensors(
        db_path: str,
        table_name: str,
        keys: List[str] = ['layer', 'pooling_method', 'tensor_dim', 'tensor'],
        sql_where: Optional[str] = None,
        ) -> List[Any]:
    """Select and return all tensors from the specified SQLite database and table.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to query.
        keys (List[str]): List of keys to select from the database.
        sql_where (str): Optional SQL WHERE clause to filter results.

    Returns:
        List[Any]: A list of tensors retrieved from the database.
    """
    if 'tensor' not in keys:
        logging.warning("'tensor' key should be included to retrieve tensors; automatically adding it.")
        keys.append('tensor')
    final_results = []
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        query = f'SELECT {", ".join(keys)} FROM {table_name}'
        if sql_where:
            assert sql_where.strip().lower().startswith('where'), "sql_where should start with 'WHERE'"
            query += f' {sql_where}'
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            result_item = {key: value for key, value in zip(keys, row)}
            result_item['tensor'] = torch.load(io.BytesIO(result_item['tensor']), map_location='cpu')
            final_results.append(result_item)
    return final_results


def merge_databases(source_dbs: List[str], target_db: str, table_name: str = 'tensors') -> int:
    """Merge multiple SQLite databases into a single target database.

    This function copies all rows from source databases into the target database.
    The target database will be created if it doesn't exist.

    Args:
        source_dbs: List of source database file paths
        target_db: Target database file path
        table_name: Name of the table to merge (default: 'tensors')

    Returns:
        Total number of rows in merged database

    Raises:
        sqlite3.Error: If database operations fail
    """
    logging.info(f"Merging {len(source_dbs)} databases into {target_db}")

    # Create or connect to target database
    target_conn = sqlite3.connect(target_db)
    target_cursor = target_conn.cursor()

    # Get table schema from first source database
    if source_dbs and os.path.exists(source_dbs[0]):
        source_conn = sqlite3.connect(source_dbs[0])
        source_cursor = source_conn.cursor()

        # Get CREATE TABLE statement
        source_cursor.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        result = source_cursor.fetchone()

        if result:
            create_statement = result[0]
            # Create table if it doesn't exist
            target_cursor.execute(create_statement)

        source_conn.close()

    # Merge each source database
    for source_db in source_dbs:
        if not os.path.exists(source_db):
            logging.warning(f"Source database {source_db} not found, skipping")
            continue

        logging.info(f"Merging {source_db}...")

        # Attach source database
        target_cursor.execute(f"ATTACH DATABASE '{source_db}' AS source_db")

        # Get column names (excluding auto-increment id)
        target_cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in target_cursor.fetchall() if row[1] != 'id']
        columns_str = ', '.join(columns)

        # Copy all rows
        target_cursor.execute(f"""
            INSERT INTO {table_name} ({columns_str})
            SELECT {columns_str}
            FROM source_db.{table_name}
        """)

        # Detach source database
        target_cursor.execute("DETACH DATABASE source_db")

        target_conn.commit()

    # Get final count
    target_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = target_cursor.fetchone()[0]

    target_conn.close()

    logging.info(f"Merge complete: {total_rows} total rows in {target_db}")
    return total_rows
