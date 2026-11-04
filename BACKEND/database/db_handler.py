"""SQLite prediction metadata. Raw input and cleaned text are never persisted."""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DATABASE_PATH = Path(os.environ.get(
    "ELARIA_DATABASE_PATH",
    str(Path(__file__).resolve().parent / "elaria.db"),
)).expanduser().resolve()


def initialize_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                source TEXT NOT NULL,
                prediction TEXT NOT NULL,
                model_score REAL,
                created_at TEXT NOT NULL
            )
        """)


def save_predictions(records):
    """Insert the entire batch in one transaction; roll back on any failure."""
    ids = []
    with closing(sqlite3.connect(DATABASE_PATH)) as connection, connection:
        for record in records:
            cursor = connection.execute("""
                INSERT INTO predictions
                    (batch_id, source, prediction, model_score, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record.get("batch_id"), record["source"], record["prediction"],
                record["model_score"], datetime.now(timezone.utc).isoformat(),
            ))
            ids.append(cursor.lastrowid)
    return ids


def save_prediction(prediction, model_score, source, batch_id=None):
    return save_predictions([{
        "prediction": prediction, "model_score": model_score,
        "source": source, "batch_id": batch_id,
    }])[0]


def get_prediction_history(limit=100):
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT id, batch_id, source, prediction, model_score, created_at
            FROM predictions ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_prediction_count():
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        return connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
