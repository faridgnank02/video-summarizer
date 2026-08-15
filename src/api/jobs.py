# src/api/jobs.py
"""SQLite job store for the API adapter."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.video_intelligence.schemas import AnalysisReport, JobOptions, VideoSource


class JobStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    report_json TEXT,
                    error TEXT,
                    trace_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, job_id: str, source: VideoSource, options: JobOptions) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, source_json, options_json) VALUES (?, 'queued', ?, ?)",
                (job_id, source.model_dump_json(), options.model_dump_json()),
            )

    def update(self, job_id: str, status: str | None = None,
               report: AnalysisReport | None = None, error: str | None = None,
               trace_id: str | None = None) -> None:
        sets, vals = [], []
        if status is not None:
            sets.append("status = ?"); vals.append(status)
        if report is not None:
            sets.append("report_json = ?"); vals.append(report.model_dump_json())
        if error is not None:
            sets.append("error = ?"); vals.append(error)
        if trace_id is not None:
            sets.append("trace_id = ?"); vals.append(trace_id)
        if not sets:
            return
        vals.append(job_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)

    def get(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["id"],
            "status": row["status"],
            "source": json.loads(row["source_json"]),
            "options": json.loads(row["options_json"]),
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
            "error": row["error"],
            "trace_id": row["trace_id"],
        }
