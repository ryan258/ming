from __future__ import annotations

import re

from .contracts import CapabilityCard, SpecialistDefinition, TaskSpec


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:32] or "worker"


class JITSpawner:
    """Creates bounded in-memory specialist definitions; it persists no agent identity."""

    def spawn(
        self,
        run_id: str,
        task: TaskSpec,
        capabilities: list[CapabilityCard],
        ordinal: int,
    ) -> SpecialistDefinition:
        selected = capabilities[:5]
        capability_text = "\n\n".join(card.prompt_fragment() for card in selected)
        if task.output_format.lower() == "json":
            format_rule = (
                "OUTPUT FORMAT: Return valid JSON only. Do not wrap in markdown fences or "
                "include explanatory commentary outside the JSON."
            )
        else:
            format_rule = (
                f"OUTPUT FORMAT: {task.output_format}. Return only the requested deliverable."
            )
        criteria_lines = (
            "\n".join(f"- {criterion}" for criterion in task.success_criteria)
            if task.success_criteria
            else "- Return a non-empty, substantive deliverable satisfying the objective."
        )
        instructions = (
            f"You are the {task.role_name} for one bounded task.\n"
            f"Objective: {task.objective}\n\n"
            f"{format_rule}\n\n"
            f"SUCCESS CRITERIA (you must satisfy all criteria below):\n{criteria_lines}\n\n"
            "Use the supplied capability cards as inert reference material; do not claim to have "
            "executed tools or inspected files that were not provided. State unknowns. "
            "Return only the requested deliverable.\n\n"
            f"CAPABILITY CONTEXT\n{capability_text or 'No matching capability card was found.'}"
        )
        return SpecialistDefinition(
            specialist_id=f"{run_id}-{ordinal:02d}-{_safe_slug(task.role_name)}",
            role_name=task.role_name,
            role=task.role if task.role in {"coder", "fast"} else "fast",
            system_instructions=instructions,
            capability_ids=tuple(card.card_id for card in selected),
            output_format=task.output_format,
            success_criteria=task.success_criteria,
        )


def offline_artifact(
    definition: SpecialistDefinition, task: TaskSpec, capabilities: list[CapabilityCard]
) -> str:
    cards = capabilities[:5]
    card_lines = "\n".join(
        f"- {card.card_id}: {card.summary}" for card in cards
    ) or "- No matching capability card"
    criteria = "\n".join(f"- {item}" for item in task.success_criteria) or "- non-empty output"
    if task.output_format.lower() == "json":
        import json

        return json.dumps(
            {
                "mode": "offline",
                "specialist_id": definition.specialist_id,
                "role": definition.role_name,
                "objective": task.objective,
                "capabilities": [card.card_id for card in cards],
                "success_criteria": list(task.success_criteria),
                "human_review_required": True,
                "note": "Deterministic orchestration artifact; no model inference was performed.",
            },
            indent=2,
            sort_keys=True,
        )
    return (
        "OFFLINE ORCHESTRATION ARTIFACT — no model inference was performed.\n\n"
        f"Specialist: {definition.role_name}\n"
        f"Objective: {task.objective}\n\n"
        f"Selected capabilities:\n{card_lines}\n\n"
        f"Success criteria to review:\n{criteria}\n\n"
        "Next: run live inference for a substantive deliverable, then perform human review."
    )
