# common_utils/db_utils.py

import sqlite3
import pandas as pd
from pathlib import Path


class DatabaseConnector:
    """
    Utility class to handle database connections and queries.
    Works for all regression problems using Regression.db
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

    def _connect(self):
        """Create a database connection"""
        return sqlite3.connect(self.db_path)

    def fetch_table(self, table_name: str, limit: int = None) -> pd.DataFrame:
        """
        Fetch full table or limited records from database.

        Args:
            table_name (str): Name of the table
            limit (int, optional): Number of records to fetch

        Returns:
            pd.DataFrame
        """
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn)

        return df

    def fetch_split_data(
        self,
        table_name: str,
        train_size: int,
        val_size: int
    ):
        """
        Fetch train, validation and live data splits sequentially.

        Args:
            table_name (str)
            train_size (int)
            val_size (int)

        Returns:
            train_df, val_df, live_df
        """

        query = f"""
        SELECT * FROM {table_name}
        """

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn)

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        live_df = df.iloc[train_size + val_size:]

        return train_df, val_df, live_df
