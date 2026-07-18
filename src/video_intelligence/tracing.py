"""SQLite-backed per-stage trace spans (cost, latency, model choice)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .schemas import TraceSpan


class TraceStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    span_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def add_span(self, trace_id: str, span: TraceSpan) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO spans (trace_id, span_json) VALUES (?, ?)",
                (trace_id, span.model_dump_json()),
            )

    def spans(self, trace_id: str) -> list[TraceSpan]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT span_json FROM spans WHERE trace_id = ? ORDER BY id", (trace_id,)
            ).fetchall()
        return [TraceSpan.model_validate_json(row[0]) for row in rows]

    def total_cost(self, trace_id: str) -> float:
        return round(sum(s.cost_usd for s in self.spans(trace_id)), 6)
