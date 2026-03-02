import sqlite3
import os
import json
from typing import Dict, Any

class EvaluationCache:
    """
    Integrates basic localized caching logic via SQLite for historical evaluations and weight tracking.
    """
    def __init__(self, db_path: str = "data/chimera_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._initialize_db()

    def _initialize_db(self):
        cursor = self.conn.cursor()
        # Table for storing backtest execution summaries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                params TEXT,
                start_time INTEGER,
                end_time INTEGER,
                total_return REAL,
                max_drawdown REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Table for storing AI weights metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                loss REAL,
                filepath TEXT NOT NULL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def log_backtest(self, strategy_name: str, params: Dict[str, Any], start_time: int, end_time: int, total_return: float, max_drawdown: float):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO backtests (strategy_name, params, start_time, end_time, total_return, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (strategy_name, json.dumps(params), start_time, end_time, total_return, max_drawdown))
        self.conn.commit()
        return cursor.lastrowid

    def log_ai_weights(self, model_name: str, loss: float, filepath: str, metadata: Dict[str, Any] = None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO ai_weights (model_name, loss, filepath, metadata)
            VALUES (?, ?, ?, ?)
        ''', (model_name, loss, filepath, json.dumps(metadata) if metadata else None))
        self.conn.commit()
        return cursor.lastrowid
    
    def close(self):
        self.conn.close()
