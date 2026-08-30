from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path

from codex.indexer import CodexIndex
from engine.config import Settings
from engine.llm import (
    AllModelsFailedError,
    MissingAPIKeyError,
    OpenRouterGateway,
    TransportResponse,
)
from engine.orchestrator import Orchestrator
from engine.output_store import IterationStore
from engine.telemetry import TelemetryStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, tmp: str) -> Orchestrator:
        codex = CodexIndex(Path(tmp) / "codex.db", PROJECT_ROOT / "codex" / "tools")
        codex.rebuild()
        telemetry = TelemetryStore(Path(tmp) / "telemetry.db")
        outputs = Path(tmp) / "outputs"
        return Orchestrator(
            Settings(data_dir=Path(tmp), output_dir=outputs),
            codex,
            telemetry,
            iterations=IterationStore(outputs / "iterations"),
        )

    async def test_iteration_snapshot_failure_does_not_mark_completed_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)

            def broken_write(*args: object, **kwargs: object) -> Path:
                raise OSError("disk full")

            assert orchestrator.iterations is not None
            orchestrator.iterations.write = broken_write  # type: ignore[method-assign]
            report = await orchestrator.run("Summarize input", offline=True)
            runs = orchestrator.telemetry.summary()["runs"]
        self.assertEqual(report.status, "completed_offline")
        self.assertIsNone(report.iteration_path)
        self.assertEqual(runs[0]["status"], "completed_offline")

    async def test_offline_run_retrieves_capability_and_records_no_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            report = await orchestrator.run(
                "Review a Python handler for path traversal", offline=True
            )
            summary = orchestrator.telemetry.summary()
            iteration = Path(report.iteration_path or "")
            self.assertTrue(iteration.is_file())
        self.assertEqual(report.status, "completed_offline")
        self.assertEqual(report.total_tokens, 0)
        self.assertIn("security.python-review", {c.card_id for c in report.capabilities})
        self.assertEqual(summary["calls"], 0)
        self.assertTrue(all(result.offline for result in report.results))

    async def test_offline_json_is_real_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = await self.make_runtime(tmp).run(
                "Create a deterministic JSON summary", offline=True, output_format="json"
            )
        passed, total = report.validation_counts
        self.assertEqual((passed, total), (2, 2))

    async def test_offline_does_not_fake_success_criterion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = await self.make_runtime(tmp).run(
                "Design an API",
                offline=True,
                success_criteria=("mentions deployment proof",),
            )
        self.assertEqual(report.status, "partial")
        criterion = report.results[0].validations[-1]
        self.assertFalse(criterion.passed)
        self.assertIn("offline mode", criterion.evidence)

    async def test_estimated_value_is_labeled_and_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = await self.make_runtime(tmp).run(
                "Summarize input", offline=True, estimated_hours=2
            )
        data = report.to_dict()
        self.assertEqual(data["estimate"]["comparison_value"], 150)
        self.assertIn("not measured", data["estimate"]["label"])

    async def test_events_in_report_are_scoped_to_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            first = await orchestrator.run("First summary", offline=True)
            second = await orchestrator.run("Second summary", offline=True)
        self.assertTrue(first.events)
        self.assertTrue(second.events)
        self.assertTrue(all(event.run_id == second.run_id for event in second.events))

    async def test_post_plan_failure_marks_run_failed_not_left_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            real_search = orchestrator.codex.search
            calls = {"n": 0}

            def flaky_search(*args: object, **kwargs: object):
                calls["n"] += 1
                if calls["n"] >= 2:  # the per-task search, after planning succeeds
                    raise RuntimeError("codex exploded mid-run")
                return real_search(*args, **kwargs)

            orchestrator.codex.search = flaky_search  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError):
                await orchestrator.run("Summarize input", offline=True)
            runs = orchestrator.telemetry.summary()["runs"]
        self.assertEqual(runs[0]["status"], "failed")
        self.assertIsNotNone(runs[0]["completed_at"])

    async def test_run_marked_failed_even_when_event_sink_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)

            def broken_record_event(_event: object) -> None:
                raise RuntimeError("event sink failed")

            orchestrator.telemetry.record_event = broken_record_event  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError):
                await orchestrator.run("Summarize input", offline=True)
            runs = orchestrator.telemetry.summary()["runs"]
        self.assertEqual(runs[0]["status"], "failed")
        self.assertIsNotNone(runs[0]["completed_at"])

    async def test_live_failure_is_recorded_as_failed_not_left_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            with self.assertRaises(MissingAPIKeyError):
                await orchestrator.run("Create a summary", offline=False)
            runs = orchestrator.telemetry.summary()["runs"]
            failed_iterations = list((Path(tmp) / "outputs" / "iterations").glob("*.json"))
        self.assertEqual(runs[0]["status"], "failed")
        self.assertIsNotNone(runs[0]["completed_at"])
        self.assertEqual(len(failed_iterations), 1)

    async def test_live_specialist_receives_output_format_and_criteria_in_prompt(self) -> None:
        recorded_calls: list[dict[str, object]] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            data = json.loads(payload)
            recorded_calls.append(data)
            # First call is planner
            if len(recorded_calls) == 1:
                return TransportResponse(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "tasks": [
                                                {
                                                    "role_name": "API Specialist",
                                                    "objective": "Build API deliverable",
                                                    "role": "fast",
                                                    "output_format": "json",
                                                    "success_criteria": [
                                                        "mentions deployment proof"
                                                    ],
                                                }
                                            ]
                                        }
                                    )
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    },
                )
            # Second call is specialist
            return TransportResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "status": "ok",
                                        "detail": "mentions deployment proof",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 15,
                        "completion_tokens": 25,
                        "total_tokens": 40,
                    },
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            codex = CodexIndex(Path(tmp) / "codex.db", PROJECT_ROOT / "codex" / "tools")
            codex.rebuild()
            telemetry = TelemetryStore(Path(tmp) / "telemetry.db")
            outputs = Path(tmp) / "outputs"
            settings = Settings(
                api_key="test-key",
                data_dir=Path(tmp),
                output_dir=outputs,
                max_retries=0,
            )
            gateway = OpenRouterGateway(settings, transport=transport)
            orchestrator = Orchestrator(
                settings,
                codex,
                telemetry,
                gateway=gateway,
                iterations=IterationStore(outputs / "iterations"),
            )
            report = await orchestrator.run(
                "Build API deliverable",
                offline=False,
                output_format="json",
                success_criteria=("mentions deployment proof",),
            )

        self.assertEqual(len(recorded_calls), 2)
        specialist_payload = recorded_calls[1]
        messages = specialist_payload["messages"]  # type: ignore[index]
        system_content = messages[0]["content"]  # type: ignore[index]
        json_instruction_present = "JSON" in system_content and "OUTPUT FORMAT" in system_content
        criterion_present = "mentions deployment proof" in system_content

        self.assertTrue(json_instruction_present)
        self.assertTrue(criterion_present)
        self.assertEqual(report.status, "completed")
        passed, total = report.validation_counts
        self.assertEqual(passed, total)
        self.assertGreater(passed, 0)

    async def test_live_planner_schema_uses_configured_max_specialists(self) -> None:
        captured: dict[str, object] = {}

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            data = json.loads(payload)
            if "response_format" in data and "maxItems" not in captured:
                schema = data["response_format"]["json_schema"]["schema"]
                captured["maxItems"] = schema["properties"]["tasks"]["maxItems"]
                captured["system"] = data["messages"][0]["content"]
            return TransportResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "tasks": [
                                            {
                                                "role_name": "Task Specialist",
                                                "objective": "Summarize input",
                                                "role": "fast",
                                                "output_format": "text",
                                                "success_criteria": [],
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            orchestrator.settings = Settings(
                api_key="test-key",
                data_dir=Path(tmp),
                output_dir=Path(tmp) / "outputs",
                max_retries=0,
                max_specialists=2,
            )
            orchestrator.gateway = OpenRouterGateway(
                orchestrator.settings, transport=transport
            )
            report = await orchestrator.run("Summarize input", offline=False)
        self.assertEqual(captured["maxItems"], 2)
        self.assertIn("at most 2", captured["system"])  # type: ignore[arg-type]
        self.assertEqual(report.status, "completed")

    async def test_live_plan_accepts_markdown_fenced_json(self) -> None:
        calls = 0
        inner = json.dumps(
            {
                "tasks": [
                    {
                        "role_name": "Task Specialist",
                        "objective": "Summarize input",
                        "role": "fast",
                        "output_format": "text",
                        "success_criteria": [],
                    }
                ]
            }
        )

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            nonlocal calls
            calls += 1
            content = f"```json\n{inner}\n```" if calls == 1 else "done"
            return TransportResponse(
                200,
                {"choices": [{"message": {"content": content}}]},
            )

        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            orchestrator.gateway = OpenRouterGateway(
                Settings(
                    api_key="test-key",
                    data_dir=Path(tmp),
                    output_dir=Path(tmp) / "outputs",
                    max_retries=0,
                ),
                transport=transport,
            )
            report = await orchestrator.run("Summarize input", offline=False)
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.results[0].definition.role_name, "Task Specialist")

    async def test_live_model_fallback_is_recorded_in_report_and_audit_database(self) -> None:
        calls: list[dict[str, object]] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            data = json.loads(payload)
            calls.append(data)
            # Planner call
            if len(calls) == 1:
                return TransportResponse(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "tasks": [
                                                {
                                                    "role_name": "Task Specialist",
                                                    "objective": "Process data",
                                                    "role": "fast",
                                                    "output_format": "text",
                                                    "success_criteria": ["mentions completed"],
                                                }
                                            ]
                                        }
                                    )
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                    },
                )
            # Specialist calls: first model fails with 404, fallback succeeds
            model = data["model"]
            if model != "openrouter/free":
                return TransportResponse(
                    404, {"error": {"message": "primary model unavailable"}}
                )
            return TransportResponse(
                200,
                {
                    "model": "openrouter/free",
                    "choices": [{"message": {"content": "Task mentions completed successfully."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            codex = CodexIndex(Path(tmp) / "codex.db", PROJECT_ROOT / "codex" / "tools")
            codex.rebuild()
            telemetry = TelemetryStore(Path(tmp) / "telemetry.db")
            outputs = Path(tmp) / "outputs"
            settings = Settings(
                api_key="test-key",
                data_dir=Path(tmp),
                output_dir=outputs,
                fast_models=("openai/gpt-oss-20b:free", "openrouter/free"),
                max_retries=0,
            )
            gateway = OpenRouterGateway(settings, transport=transport)
            orchestrator = Orchestrator(
                settings,
                codex,
                telemetry,
                gateway=gateway,
                iterations=IterationStore(outputs / "iterations"),
            )
            report = await orchestrator.run(
                "Process data",
                offline=False,
                output_format="text",
                success_criteria=("mentions completed",),
            )
            db_calls = telemetry.calls_for_run(report.run_id)

        # Verify fallback is exposed in report
        self.assertTrue(report.fallback_summary)
        observed_fallbacks = report.to_dict()["observed"]["fallbacks"]
        self.assertTrue(observed_fallbacks)
        self.assertIn("openai/gpt-oss-20b:free", observed_fallbacks[0])
        self.assertIn("HTTP 404", observed_fallbacks[0])
        self.assertIn("openrouter/free", observed_fallbacks[0])

        # Verify fallback is recorded in audit database
        self.assertEqual(len(db_calls), 2)  # orchestrator + specialist
        specialist_call = db_calls[1]
        self.assertEqual(specialist_call["attempts"], 2)
        trace = json.loads(specialist_call["attempt_trace_json"])
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["route"], "openai/gpt-oss-20b:free")
        self.assertEqual(trace[0]["outcome"], "failed")
        self.assertIn("HTTP 404", trace[0]["status_or_error"])
        self.assertEqual(trace[1]["route"], "openrouter/free")
        self.assertEqual(trace[1]["outcome"], "success")

    async def test_same_route_retry_is_reported_as_retry_not_fallback(self) -> None:
        calls: list[dict[str, object]] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            data = json.loads(payload)
            calls.append(data)
            # Planner call
            if len(calls) == 1:
                return TransportResponse(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "tasks": [
                                                {
                                                    "role_name": "Task Specialist",
                                                    "objective": "Process data",
                                                    "role": "fast",
                                                    "output_format": "text",
                                                    "success_criteria": ["mentions completed"],
                                                }
                                            ]
                                        }
                                    )
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                    },
                )
            # Specialist call 1: returns 429
            if len(calls) == 2:
                return TransportResponse(
                    429, {"error": {"message": "rate limited; please wait"}}
                )
            # Specialist call 2: retry succeeds on same model
            return TransportResponse(
                200,
                {
                    "model": data["model"],
                    "choices": [{"message": {"content": "mentions completed"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            codex = CodexIndex(Path(tmp) / "codex.db", PROJECT_ROOT / "codex" / "tools")
            codex.rebuild()
            telemetry = TelemetryStore(Path(tmp) / "telemetry.db")
            outputs = Path(tmp) / "outputs"
            settings = Settings(
                api_key="test-key",
                data_dir=Path(tmp),
                output_dir=outputs,
                fast_models=("openai/gpt-oss-20b:free", "openrouter/free"),
                max_retries=1,
            )
            gateway = OpenRouterGateway(settings, transport=transport)
            orchestrator = Orchestrator(
                settings,
                codex,
                telemetry,
                gateway=gateway,
                iterations=IterationStore(outputs / "iterations"),
            )
            report = await orchestrator.run(
                "Process data",
                offline=False,
                output_format="text",
                success_criteria=("mentions completed",),
            )

        self.assertTrue(report.fallback_summary)
        summary_text = report.fallback_summary[0]
        self.assertIn("retried openai/gpt-oss-20b:free once after HTTP 429", summary_text)
        self.assertIn("succeeded", summary_text)
        self.assertNotIn("fell back to", summary_text)

    async def test_terminal_all_routes_failure_preserves_attempt_trace_in_telemetry_and_iteration(
        self,
    ) -> None:
        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            # Planner fails for all routes
            return TransportResponse(
                503, {"error": {"message": "service temporarily unavailable"}}
            )

        with tempfile.TemporaryDirectory() as tmp:
            codex = CodexIndex(Path(tmp) / "codex.db", PROJECT_ROOT / "codex" / "tools")
            codex.rebuild()
            telemetry = TelemetryStore(Path(tmp) / "telemetry.db")
            outputs = Path(tmp) / "outputs"
            settings = Settings(
                api_key="test-key",
                data_dir=Path(tmp),
                output_dir=outputs,
                orchestrator_models=("nvidia/nemotron-3-ultra:free", "openrouter/free"),
                max_retries=0,
            )
            gateway = OpenRouterGateway(settings, transport=transport)
            orchestrator = Orchestrator(
                settings,
                codex,
                telemetry,
                gateway=gateway,
                iterations=IterationStore(outputs / "iterations"),
            )
            with self.assertRaises(AllModelsFailedError) as cm:
                await orchestrator.run("Plan my project", offline=False)

            exc = cm.exception
            self.assertTrue(len(exc.attempt_trace) >= 2)
            self.assertEqual(exc.attempt_trace[0].route, "nvidia/nemotron-3-ultra:free")
            self.assertEqual(exc.attempt_trace[1].route, "openrouter/free")

            runs = telemetry.summary()["runs"]
            self.assertEqual(runs[0]["status"], "failed")

            # Telemetry call record exists and has attempt trace
            db_calls = telemetry.calls_for_run(runs[0]["run_id"])
            self.assertEqual(len(db_calls), 1)
            trace = json.loads(db_calls[0]["attempt_trace_json"])
            self.assertEqual(len(trace), 2)
            self.assertEqual(trace[0]["route"], "nvidia/nemotron-3-ultra:free")
            self.assertEqual(trace[0]["outcome"], "failed")

            # Iteration record exists and has attempt trace
            iteration_files = list((outputs / "iterations").glob("*.json"))
            self.assertEqual(len(iteration_files), 1)
            iteration_data = json.loads(iteration_files[0].read_text())
            self.assertEqual(iteration_data["status"], "failed")
            self.assertIn("saved_attempt_trace", iteration_data)
            self.assertTrue(len(iteration_data["saved_attempt_trace"]) >= 2)
            self.assertIn("attempt_trace", iteration_data["error"])

    async def test_live_plan_malformed_task_is_valueerror_and_marks_failed(self) -> None:
        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            return TransportResponse(
                200,
                {
                    "model": "nvidia/nemotron-3-ultra:free",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "tasks": [
                                            {
                                                "role_name": "API Specialist",
                                                "objective": "Build API deliverable",
                                                "role": "fast",
                                                "output_format": "json",
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 59,
                        "total_tokens": 99,
                    },
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            orchestrator.gateway = OpenRouterGateway(
                Settings(
                    api_key="test-key",
                    data_dir=Path(tmp),
                    output_dir=Path(tmp) / "outputs",
                    max_retries=0,
                ),
                transport=transport,
            )
            with self.assertRaisesRegex(ValueError, "invalid structured task JSON"):
                await orchestrator.run("Build API deliverable", offline=False)
            runs = orchestrator.telemetry.summary()["runs"]
            db_calls = orchestrator.telemetry.calls_for_run(runs[0]["run_id"])
        self.assertEqual(runs[0]["status"], "failed")
        self.assertEqual(len(db_calls), 1)
        self.assertEqual(db_calls[0]["specialist_id"], "orchestrator")
        self.assertEqual(db_calls[0]["actual_model"], "nvidia/nemotron-3-ultra:free")
        self.assertEqual(db_calls[0]["total_tokens"], 99)

    async def test_cancelled_run_is_marked_cancelled_not_left_running(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test gate timed out")
            return TransportResponse(503, {"error": {"message": "slow"}})

        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self.make_runtime(tmp)
            orchestrator.gateway = OpenRouterGateway(
                Settings(
                    api_key="test-key",
                    data_dir=Path(tmp),
                    output_dir=Path(tmp) / "outputs",
                    max_retries=0,
                ),
                transport=transport,
            )
            task = asyncio.create_task(orchestrator.run("Plan my project", offline=False))
            await asyncio.to_thread(started.wait, 5)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            runs = orchestrator.telemetry.summary()["runs"]
        self.assertEqual(runs[0]["status"], "cancelled")
        self.assertIsNotNone(runs[0]["completed_at"])


if __name__ == "__main__":
    unittest.main()
