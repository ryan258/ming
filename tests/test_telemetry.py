from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.contracts import ModelResponse, Usage
from engine.telemetry import TelemetryStore


class TelemetryTests(unittest.TestCase):
    def test_call_summary_uses_observed_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TelemetryStore(Path(tmp) / "telemetry.db")
            store.start_run("run-1", "brief", "live", "2026-01-01T00:00:00Z")
            store.record_call(
                "run-1",
                "worker",
                ModelResponse(
                    content="ok",
                    requested_model="a:free",
                    actual_model="b:free",
                    usage=Usage(2, 3, 5),
                    latency_ms=25,
                ),
            )
            summary = store.summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["tokens"], 5)
        self.assertEqual(summary["average_latency_ms"], 25)


if __name__ == "__main__":
    unittest.main()
