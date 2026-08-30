from __future__ import annotations

import json
from pathlib import Path

from codex.indexer import CodexIndex
from engine.config import Settings
from engine.orchestrator import Orchestrator
from engine.output_store import IterationStore, resolve_output_root, resolve_workspace_dir
from engine.telemetry import TelemetryStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def run_evals(settings: Settings, *, as_json: bool = False) -> int:
    data_dir = resolve_workspace_dir(settings.data_dir, Path.cwd(), env_name="MING_DATA_DIR")
    codex = CodexIndex(data_dir / "codex.db", PROJECT_ROOT / "codex" / "tools")
    codex.rebuild()
    output_dir = resolve_output_root(settings.output_dir, Path.cwd())
    telemetry = TelemetryStore(output_dir / "logs" / "telemetry.db")
    iterations = IterationStore(output_dir / "iterations")
    cases = json.loads((Path(__file__).with_name("test_cases.json")).read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    for case in cases:
        orchestrator = Orchestrator(settings, codex, telemetry, iterations=iterations)
        report = await orchestrator.run(
            case["brief"], offline=True, output_format=case["output_format"]
        )
        retrieved = {card.card_id for card in report.capabilities}
        capability_passed = case["expected_capability"] in retrieved
        passed, total = report.validation_counts
        results.append(
            {
                "id": case["id"],
                "passed": capability_passed and passed == total,
                "expected_capability": case["expected_capability"],
                "retrieved": sorted(retrieved),
                "validation": f"{passed}/{total}",
            }
        )
    payload = {
        "passed": sum(bool(item["passed"]) for item in results),
        "total": len(results),
        "mode": "offline deterministic",
        "results": results,
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"EVAL: {payload['passed']}/{payload['total']} deterministic fixtures passed")
        for item in results:
            mark = "PASS" if item["passed"] else "FAIL"
            print(
                f"{mark} {item['id']} — retrieval {item['expected_capability']}, "
                f"validation {item['validation']}"
            )
    return 0 if payload["passed"] == payload["total"] else 1
