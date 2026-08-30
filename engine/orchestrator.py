from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import uuid
from dataclasses import dataclass
from typing import Any

from codex.indexer import CodexIndex

from .bus import EventBus
from .config import Settings
from .contracts import (
    Event,
    ModelResponse,
    RouteAttempt,
    RunReport,
    SpecialistResult,
    TaskSpec,
    Usage,
    utc_now,
)
from .jit_spawner import JITSpawner, offline_artifact
from .llm import OpenRouterGateway
from .output_store import IterationStore
from .telemetry import TelemetryStore
from .validation import validate_output

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "role_name": {"type": "string"},
                    "objective": {"type": "string"},
                    "role": {"type": "string", "enum": ["coder", "fast"]},
                    "output_format": {"type": "string", "enum": ["text", "json"]},
                    "success_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                    },
                },
                "required": [
                    "role_name",
                    "objective",
                    "role",
                    "output_format",
                    "success_criteria",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}


def _loads_json_object(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def planner_schema(max_specialists: int) -> dict[str, Any]:
    schema = copy.deepcopy(PLANNER_SCHEMA)
    schema["properties"]["tasks"]["maxItems"] = max_specialists
    return schema


def _task_spec_from_planner(
    index: int,
    raw: object,
    success_criteria: tuple[str, ...],
    output_format: str,
) -> TaskSpec:
    if not isinstance(raw, dict):
        raise TypeError("planner task must be an object")
    role_name = raw["role_name"]
    objective = raw["objective"]
    role = raw["role"]
    planner_format = raw["output_format"]
    extra_criteria = raw["success_criteria"]
    if not isinstance(role_name, str) or not role_name.strip():
        raise TypeError("planner role_name must be a non-empty string")
    if not isinstance(objective, str) or not objective.strip():
        raise TypeError("planner objective must be a non-empty string")
    if role not in {"coder", "fast"}:
        raise TypeError("planner role must be 'coder' or 'fast'")
    if planner_format not in {"text", "json"}:
        raise TypeError("planner output_format must be 'text' or 'json'")
    if not isinstance(extra_criteria, list) or not all(
        isinstance(item, str) for item in extra_criteria
    ):
        raise TypeError("planner success_criteria must be an array of strings")
    combined_criteria = tuple(dict.fromkeys((*success_criteria, *extra_criteria)))
    return TaskSpec(
        task_id=f"task-{index:02d}",
        role_name=role_name[:80],
        objective=objective[:2000],
        role=role,
        output_format=output_format if output_format == "json" else planner_format,
        success_criteria=combined_criteria[:5],
    )


class PlannerOutputError(ValueError):
    def __init__(self, message: str, *, response: ModelResponse) -> None:
        super().__init__(message)
        self.response = response


@dataclass(slots=True)
class Plan:
    tasks: list[TaskSpec]
    response: ModelResponse | None = None


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        codex: CodexIndex,
        telemetry: TelemetryStore,
        *,
        gateway: OpenRouterGateway | None = None,
        bus: EventBus | None = None,
        iterations: IterationStore | None = None,
    ) -> None:
        self.settings = settings
        self.codex = codex
        self.telemetry = telemetry
        self.gateway = gateway or OpenRouterGateway(settings)
        self.bus = bus or EventBus()
        self.iterations = iterations
        self.spawner = JITSpawner()

    async def _event(
        self, run_id: str, kind: str, message: str, **payload: object
    ) -> None:
        event = Event(kind=kind, run_id=run_id, message=message, payload=dict(payload))
        await self.bus.publish(event)
        self.telemetry.record_event(event)

    async def _mark_failed(
        self,
        run_id: str,
        exc: BaseException,
        *,
        brief: str,
        mode: str,
        started_at: str,
        failed_specialist_id: str | None = None,
        status: str = "failed",
    ) -> None:
        completed_at = utc_now()
        attempt_trace: list[RouteAttempt] = getattr(exc, "attempt_trace", [])
        trace_dicts = [a.to_dict() for a in attempt_trace] if attempt_trace else None
        event_kind = "run.cancelled" if status == "cancelled" else "run.failed"
        event_message = (
            "run cancelled before completion"
            if status == "cancelled"
            else "run stopped before completion"
        )

        # Terminal audit record: If the gateway recorded route attempts before failing,
        # persist a failed-call record in telemetry so audit history is not empty.
        if attempt_trace:
            with contextlib.suppress(Exception):
                first_route = attempt_trace[0].route
                self.telemetry.record_call(
                    run_id,
                    failed_specialist_id or "orchestrator",
                    ModelResponse(
                        content="",
                        requested_model=first_route,
                        actual_model="none",
                        usage=Usage(),
                        latency_ms=0.0,
                        attempts=len(attempt_trace),
                        provider="openrouter",
                        attempt_trace=list(attempt_trace),
                    ),
                )

        # Terminal audit state first, and each recovery step isolated: a crashed
        # run must reach a terminal status even when the sink that crashed it
        # (events, iteration store) fails again here. Never mask the original exception.
        with contextlib.suppress(Exception):
            self.telemetry.fail_run(run_id, completed_at, status=status)
        with contextlib.suppress(Exception):
            await self._event(
                run_id,
                event_kind,
                event_message,
                error_type=type(exc).__name__,
                attempt_trace=trace_dicts,
            )
        if self.iterations is not None:
            with contextlib.suppress(Exception):
                path = self.iterations.path_for(f"run-{run_id}")
                error_info: dict[str, Any] = {
                    "type": type(exc).__name__,
                    "message": event_message,
                }
                if trace_dicts is not None:
                    error_info["attempt_trace"] = trace_dicts
                    error_info["saved_attempt_trace"] = trace_dicts
                self.iterations.write(
                    f"run-{run_id}",
                    {
                        "run_id": run_id,
                        "brief": brief,
                        "mode": mode,
                        "status": status,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "iteration_path": str(path),
                        "error": error_info,
                        "saved_attempt_trace": trace_dicts,
                        "events": [
                            event.to_dict()
                            for event in self.bus.history
                            if event.run_id == run_id
                        ],
                    },
                )

    def _offline_plan(
        self, brief: str, success_criteria: tuple[str, ...], output_format: str
    ) -> Plan:
        lowered = brief.casefold()
        if any(word in lowered for word in ("code", "python", "api", "refactor", "test")):
            role_name, role = "Implementation Specialist", "coder"
        elif any(word in lowered for word in ("json", "schema", "extract", "summar")):
            role_name, role = "Structured Data Specialist", "fast"
        else:
            role_name, role = "Task Specialist", "fast"
        return Plan(
            tasks=[
                TaskSpec(
                    task_id="task-01",
                    role_name=role_name,
                    objective=brief,
                    role=role,
                    output_format=output_format,
                    success_criteria=success_criteria,
                )
            ]
        )

    async def _live_plan(
        self, brief: str, success_criteria: tuple[str, ...], output_format: str
    ) -> Plan:
        response = await self.gateway.complete(
            "orchestrator",
            [
                {
                    "role": "system",
                    "content": (
                        "Decompose the brief into the fewest independent specialist tasks needed. "
                        f"Use at most {self.settings.max_specialists}. "
                        "Criteria must be observable phrases or properties."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Brief: {brief}\nRequested format: {output_format}\n"
                        f"Operator criteria: {list(success_criteria)}"
                    ),
                },
            ],
            response_schema=planner_schema(self.settings.max_specialists),
            max_tokens=700,
            temperature=0.1,
        )
        try:
            payload = _loads_json_object(response.content)
            if not isinstance(payload, dict):
                raise TypeError("planner payload must be an object")
            raw_tasks = payload["tasks"]
            if not isinstance(raw_tasks, list):
                raise TypeError("planner tasks must be an array")
            tasks = [
                _task_spec_from_planner(index, raw, success_criteria, output_format)
                for index, raw in enumerate(raw_tasks[: self.settings.max_specialists], 1)
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PlannerOutputError(
                "planner returned invalid structured task JSON", response=response
            ) from exc
        if not tasks:
            raise PlannerOutputError("planner returned no tasks", response=response)
        return Plan(tasks=tasks, response=response)

    async def run(
        self,
        brief: str,
        *,
        offline: bool = False,
        success_criteria: tuple[str, ...] = (),
        output_format: str = "text",
        estimated_hours: float | None = None,
    ) -> RunReport:
        brief = brief.strip()
        if not brief:
            raise ValueError("brief must not be empty")
        if output_format not in {"text", "json"}:
            raise ValueError("output_format must be 'text' or 'json'")
        if estimated_hours is not None and estimated_hours < 0:
            raise ValueError("estimated_hours cannot be negative")

        run_id = uuid.uuid4().hex[:12]
        started_at = utc_now()
        mode = "offline" if offline else "live"
        self.telemetry.start_run(run_id, brief, mode, started_at)
        try:
            return await self._run_lifecycle(
                run_id,
                brief,
                offline=offline,
                success_criteria=success_criteria,
                output_format=output_format,
                estimated_hours=estimated_hours,
                mode=mode,
                started_at=started_at,
            )
        except asyncio.CancelledError as exc:
            await self._mark_failed(
                run_id,
                exc,
                brief=brief,
                mode=mode,
                started_at=started_at,
                status="cancelled",
            )
            raise
        except Exception as exc:
            # One failure boundary: any error after start_run() marks the run
            # failed so a crashed run is never indistinguishable from a live one.
            specialist_id = getattr(exc, "specialist_id", None)
            await self._mark_failed(
                run_id,
                exc,
                brief=brief,
                mode=mode,
                started_at=started_at,
                failed_specialist_id=specialist_id,
            )
            raise

    async def _run_lifecycle(
        self,
        run_id: str,
        brief: str,
        *,
        offline: bool,
        success_criteria: tuple[str, ...],
        output_format: str,
        estimated_hours: float | None,
        mode: str,
        started_at: str,
    ) -> RunReport:
        await self._event(run_id, "run.started", f"{mode} run started", mode=mode)

        capabilities = self.codex.search(brief, limit=5)
        await self._event(
            run_id,
            "codex.searched",
            f"selected {len(capabilities)} capability cards",
            capability_ids=[card.card_id for card in capabilities],
        )

        try:
            plan = (
                self._offline_plan(brief, success_criteria, output_format)
                if offline
                else await self._live_plan(brief, success_criteria, output_format)
            )
        except PlannerOutputError as exc:
            self.telemetry.record_call(run_id, "orchestrator", exc.response)
            raise
        if plan.response:
            self.telemetry.record_call(run_id, "orchestrator", plan.response)
        await self._event(
            run_id,
            "plan.created",
            f"created {len(plan.tasks)} bounded task(s)",
            task_ids=[task.task_id for task in plan.tasks],
        )

        results: list[SpecialistResult] = []
        for ordinal, task in enumerate(plan.tasks, 1):
            task_capabilities = self.codex.search(task.objective, limit=5) or capabilities
            definition = self.spawner.spawn(run_id, task, task_capabilities, ordinal)
            await self._event(
                run_id,
                "specialist.spawned",
                f"spawned ephemeral {definition.role_name}",
                specialist_id=definition.specialist_id,
                capabilities=list(definition.capability_ids),
            )
            if offline:
                content = offline_artifact(definition, task, task_capabilities)
                response = ModelResponse(
                    content=content,
                    requested_model="offline/deterministic",
                    actual_model="offline/deterministic",
                    usage=Usage(),
                    latency_ms=0.0,
                    provider="local",
                )
            else:
                messages = [
                    {"role": "system", "content": definition.system_instructions},
                    {"role": "user", "content": task.objective},
                ]
                try:
                    response = await self.gateway.complete(definition.role, messages)
                except Exception as exc:
                    exc.specialist_id = definition.specialist_id
                    raise
                self.telemetry.record_call(run_id, definition.specialist_id, response)
                content = response.content
            validations = validate_output(
                content,
                () if offline else task.success_criteria,
                task.output_format,
            )
            if offline:
                from .contracts import ValidationResult

                validations.extend(
                    ValidationResult(
                        criterion=criterion,
                        passed=False,
                        evidence="offline mode cannot validate substantive model-output criteria",
                    )
                    for criterion in task.success_criteria
                )
            results.append(
                SpecialistResult(
                    definition=definition,
                    content=content,
                    model=response.actual_model,
                    usage=response.usage,
                    latency_ms=response.latency_ms,
                    validations=validations,
                    offline=offline,
                    requested_model=response.requested_model,
                    attempts=response.attempts,
                    attempt_trace=list(response.attempt_trace),
                )
            )
            await self._event(
                run_id,
                "specialist.completed",
                f"completed {definition.role_name}",
                specialist_id=definition.specialist_id,
                model=response.actual_model,
                passed=sum(item.passed for item in validations),
                validations=len(validations),
            )

        all_validations = [v for result in results for v in result.validations]
        passed = all(item.passed for item in all_validations)
        status = ("completed_offline" if offline else "completed") if passed else "partial"
        completed_at = utc_now()
        estimate = (
            round(estimated_hours * self.settings.human_rate, 2)
            if estimated_hours is not None
            else None
        )
        await self._event(
            run_id,
            "run.completed",
            f"run finished with status {status}",
            status=status,
            human_review_required=True,
        )
        report = RunReport(
            run_id=run_id,
            brief=brief,
            mode=mode,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            capabilities=capabilities,
            results=results,
            events=[event for event in self.bus.history if event.run_id == run_id],
            orchestration_usage=plan.response.usage if plan.response else Usage(),
            orchestration_model=plan.response.actual_model if plan.response else None,
            orchestration_latency_ms=plan.response.latency_ms if plan.response else 0.0,
            orchestration_attempt_trace=(
                list(plan.response.attempt_trace) if plan.response else []
            ),
            estimated_hours=estimated_hours,
            estimated_labor_value=estimate,
            iteration_path=(
                str(self.iterations.path_for(f"run-{run_id}"))
                if self.iterations is not None
                else None
            ),
        )
        self.telemetry.finish_run(report)
        if self.iterations is not None:
            try:
                self.iterations.write(f"run-{run_id}", report.to_dict())
            except OSError:
                # Snapshot is audit-only. The run already finished; do not rewrite
                # telemetry to failed or hide the in-memory report.
                report.iteration_path = None
        return report
