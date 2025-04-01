import sqlite3
from datetime import datetime, timezone


class Database:
    def __init__(self, db_path: str = "data/messages.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        pass

    def close(self):
        self.conn.close()
