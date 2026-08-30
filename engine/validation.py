from __future__ import annotations

import json

from .contracts import ValidationResult


def validate_output(
    content: str, criteria: tuple[str, ...], output_format: str
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    stripped = content.strip()
    results.append(
        ValidationResult(
            criterion="non-empty output",
            passed=bool(stripped),
            evidence=f"{len(stripped)} characters returned",
        )
    )
    if output_format.lower() == "json":
        try:
            parsed = json.loads(stripped)
            results.append(
                ValidationResult(
                    criterion="valid JSON",
                    passed=True,
                    evidence=f"decoded {type(parsed).__name__}",
                )
            )
        except json.JSONDecodeError as exc:
            results.append(
                ValidationResult(
                    criterion="valid JSON",
                    passed=False,
                    evidence=f"decode error at character {exc.pos}",
                )
            )

    lowered = content.casefold()
    for criterion in criteria:
        normalized = criterion.strip()
        needle = normalized
        lowered_criterion = normalized.casefold()
        if lowered_criterion.startswith("mentions "):
            needle = normalized[9:].strip().strip("'\"")
        elif lowered_criterion.startswith("contains:"):
            needle = normalized.split(":", 1)[1].strip()
        matched = bool(needle) and needle.casefold() in lowered
        results.append(
            ValidationResult(
                criterion=normalized,
                passed=matched,
                evidence=(
                    f"found {needle!r} in output"
                    if matched
                    else f"did not find {needle!r}; semantic review remains manual"
                ),
            )
        )
    return results
