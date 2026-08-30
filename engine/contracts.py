from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import groupby
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Event:
    kind: str
    run_id: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CapabilityCard:
    card_id: str
    title: str
    summary: str
    tags: tuple[str, ...]
    body: str
    source: str
    score: float = 0.0

    def prompt_fragment(self) -> str:
        return (
            f"CAPABILITY {self.card_id}: {self.title}\n"
            f"Tags: {', '.join(self.tags)}\n{self.summary}\n{self.body.strip()}"
        )


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    role_name: str
    objective: str
    role: str = "fast"
    output_format: str = "text"
    success_criteria: tuple[str, ...] = ()


@dataclass(slots=True)
class SpecialistDefinition:
    specialist_id: str
    role_name: str
    role: str
    system_instructions: str
    capability_ids: tuple[str, ...]
    output_format: str
    success_criteria: tuple[str, ...]


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    available: bool = True


def _coerce_token(value: object) -> tuple[int, bool]:
    if value is None or value == "":
        return 0, True
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0, False
    if parsed < 0:
        return 0, False
    return parsed, True


def usage_from_provider(data: object) -> Usage:
    if not isinstance(data, dict):
        return Usage(available=False)
    prompt_tokens, prompt_ok = _coerce_token(data.get("prompt_tokens"))
    completion_tokens, completion_ok = _coerce_token(data.get("completion_tokens"))
    total_tokens, total_ok = _coerce_token(data.get("total_tokens"))
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        available=prompt_ok and completion_ok and total_ok,
    )


@dataclass(slots=True)
class RouteAttempt:
    route: str
    status_or_error: str
    retry: int
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_attempts(role_name: str, attempt_trace: Sequence[RouteAttempt]) -> list[str]:
    if not attempt_trace:
        return []

    grouped: list[tuple[str, list[RouteAttempt]]] = [
        (route, list(items)) for route, items in groupby(attempt_trace, key=lambda a: a.route)
    ]

    summaries: list[str] = []

    if len(grouped) == 1:
        route, attempts = grouped[0]
        retries = [a for a in attempts if a.outcome != "success"]
        if retries:
            last = attempts[-1]
            reasons = ", ".join(dict.fromkeys(a.status_or_error for a in retries))
            count_str = "once" if len(retries) == 1 else f"{len(retries)} times"
            status_str = "succeeded" if last.outcome == "success" else "failed"
            summaries.append(
                f"{role_name}: retried {route} {count_str} after {reasons}; {status_str}"
            )
        return summaries

    dest_route, dest_attempts = grouped[-1]
    dest_succeeded = any(a.outcome == "success" for a in dest_attempts)
    target_desc = dest_route if dest_succeeded else "none"

    for route, attempts in grouped[:-1]:
        reasons = ", ".join(dict.fromkeys(a.status_or_error for a in attempts))
        summaries.append(
            f"{role_name}: {route} failed ({reasons}) -> fell back to {target_desc}"
        )

    dest_retries = [a for a in dest_attempts if a.outcome != "success"]
    if dest_retries:
        reasons = ", ".join(dict.fromkeys(a.status_or_error for a in dest_retries))
        count_str = "once" if len(dest_retries) == 1 else f"{len(dest_retries)} times"
        status_str = "succeeded" if dest_succeeded else "failed"
        summaries.append(
            f"{role_name}: retried {dest_route} {count_str} after {reasons}; {status_str}"
        )

    return summaries


@dataclass(slots=True)
class ModelResponse:
    content: str
    requested_model: str
    actual_model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    attempts: int = 1
    provider: str = "openrouter"
    attempt_trace: list[RouteAttempt] = field(default_factory=list)


@dataclass(slots=True)
class ValidationResult:
    criterion: str
    passed: bool
    evidence: str


@dataclass(slots=True)
class SpecialistResult:
    definition: SpecialistDefinition
    content: str
    model: str
    usage: Usage
    latency_ms: float
    validations: list[ValidationResult]
    offline: bool = False
    requested_model: str | None = None
    attempts: int = 1
    attempt_trace: list[RouteAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunReport:
    run_id: str
    brief: str
    mode: str
    status: str
    started_at: str
    completed_at: str
    capabilities: list[CapabilityCard]
    results: list[SpecialistResult]
    events: list[Event]
    orchestration_usage: Usage = field(default_factory=Usage)
    orchestration_model: str | None = None
    orchestration_latency_ms: float = 0.0
    orchestration_attempt_trace: list[RouteAttempt] = field(default_factory=list)
    estimated_hours: float | None = None
    estimated_labor_value: float | None = None
    iteration_path: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.orchestration_usage.total_tokens + sum(
            result.usage.total_tokens for result in self.results
        )

    @property
    def validation_counts(self) -> tuple[int, int]:
        validations = [v for result in self.results for v in result.validations]
        return sum(item.passed for item in validations), len(validations)

    @property
    def fallback_summary(self) -> list[str]:
        fallbacks: list[str] = []
        if self.orchestration_attempt_trace:
            fallbacks.extend(summarize_attempts("orchestrator", self.orchestration_attempt_trace))
        for result in self.results:
            fallbacks.extend(summarize_attempts(result.definition.role_name, result.attempt_trace))
        return fallbacks

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        passed, total = self.validation_counts
        data["observed"] = {
            "total_tokens": self.total_tokens,
            "validation_passed": passed,
            "validation_total": total,
            "actual_models": list(dict.fromkeys(result.model for result in self.results)),
            "orchestration_model": self.orchestration_model,
            "latency_ms": round(
                self.orchestration_latency_ms + sum(result.latency_ms for result in self.results), 2
            ),
            "fallbacks": self.fallback_summary,
        }
        data["estimate"] = {
            "hours": self.estimated_hours,
            "comparison_value": self.estimated_labor_value,
            "human_rate": (
                self.estimated_labor_value / self.estimated_hours
                if self.estimated_hours and self.estimated_labor_value is not None
                else None
            ),
            "label": "operator-supplied comparison estimate, not measured savings",
        }
        return data
