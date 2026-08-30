from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import Event, ModelResponse, RunReport
from .image_generator import ImageArtifact


class TelemetryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    brief TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    estimated_hours REAL,
                    estimated_labor_value REAL
                );
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    specialist_id TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    actual_model TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    attempt_trace_json TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS image_generations (
                    generation_id TEXT PRIMARY KEY,
                    need TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    actual_model TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    authorized_max_cost_usd REAL NOT NULL,
                    observed_cost_usd REAL,
                    budget_exceeded INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            with contextlib.suppress(sqlite3.OperationalError):
                connection.execute(
                    "ALTER TABLE calls ADD COLUMN attempt_trace_json TEXT NOT NULL DEFAULT '[]'"
                )

    def start_run(self, run_id: str, brief: str, mode: str, started_at: str) -> None:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, brief, mode, status, started_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, brief, mode, "running", started_at),
            )

    def record_event(self, event: Event) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events(run_id, kind, message, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.kind,
                    event.message,
                    json.dumps(event.payload, sort_keys=True),
                    event.created_at,
                ),
            )

    def record_call(
        self, run_id: str, specialist_id: str, response: ModelResponse
    ) -> None:
        raw_trace = getattr(response, "attempt_trace", [])
        trace_json = json.dumps([a.to_dict() for a in raw_trace])
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO calls(
                    run_id, specialist_id, requested_model, actual_model, latency_ms,
                    prompt_tokens, completion_tokens, total_tokens, attempts, attempt_trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    specialist_id,
                    response.requested_model,
                    response.actual_model,
                    response.latency_ms,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                    response.attempts,
                    trace_json,
                ),
            )

    def finish_run(self, report: RunReport) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status=?, completed_at=?, estimated_hours=?, estimated_labor_value=?
                WHERE run_id=?
                """,
                (
                    report.status,
                    report.completed_at,
                    report.estimated_hours,
                    report.estimated_labor_value,
                    report.run_id,
                ),
            )

    def fail_run(
        self, run_id: str, completed_at: str, *, status: str = "failed"
    ) -> None:
        if status not in {"failed", "cancelled"}:
            status = "failed"
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status=?, completed_at=? WHERE run_id=?",
                (status, completed_at, run_id),
            )

    def record_image(self, artifact: ImageArtifact) -> None:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_generations(
                    generation_id, need, requested_model, actual_model, output_path,
                    sha256, media_type, estimated_cost_usd, authorized_max_cost_usd,
                    observed_cost_usd, budget_exceeded, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.generation_id,
                    artifact.need,
                    artifact.requested_model,
                    artifact.actual_model,
                    artifact.output_path,
                    artifact.sha256,
                    artifact.media_type,
                    artifact.estimated_cost_usd,
                    artifact.authorized_max_cost_usd,
                    artifact.observed_cost_usd,
                    int(artifact.budget_exceeded),
                    artifact.latency_ms,
                    artifact.created_at,
                ),
            )

    def summary(self, limit: int = 20) -> dict[str, Any]:
        self.initialize()
        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT count(*) AS calls, coalesce(sum(total_tokens), 0) AS tokens,
                       coalesce(avg(latency_ms), 0) AS average_latency_ms
                FROM calls
                """
            ).fetchone()
            runs = connection.execute(
                "SELECT run_id, mode, status, started_at, completed_at FROM runs "
                "ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            image_totals = connection.execute(
                """
                SELECT count(*) AS images,
                       coalesce(sum(observed_cost_usd), 0) AS observed_cost_usd
                FROM image_generations
                """
            ).fetchone()
        return {
            "calls": int(totals["calls"]),
            "tokens": int(totals["tokens"]),
            "average_latency_ms": round(float(totals["average_latency_ms"]), 2),
            "images": int(image_totals["images"]),
            "image_observed_cost_usd": round(
                float(image_totals["observed_cost_usd"]), 6
            ),
            "runs": [dict(row) for row in runs],
        }

    def calls_for_run(self, run_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM calls WHERE run_id=? ORDER BY id ASC", (run_id,)
            ).fetchall()
            return [dict(row) for row in rows]
