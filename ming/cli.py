from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import sys
import uuid
from pathlib import Path

from codex.indexer import CodexIndex
from engine.config import ConfigurationError, Settings
from engine.contracts import utc_now
from engine.image_generator import (
    SUPPORTED_ASPECT_RATIOS,
    ImageArtist,
    ImageGenerationCancelled,
    ImageRequest,
)
from engine.llm import GatewayError
from engine.orchestrator import Orchestrator
from engine.output_store import IterationStore, resolve_output_root, resolve_workspace_dir
from engine.telemetry import TelemetryStore

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PACKAGE_ROOT / "codex" / "tools"


def _finite_non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"{value!r} must be a finite number")
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{value!r} cannot be negative")
    return parsed


def _settings() -> Settings:
    return Settings.from_env(Path.cwd() / ".env")


def _workspace_root(workspace_root: Path | None = None) -> Path:
    return (workspace_root or Path.cwd()).resolve()


def _data_path(
    settings: Settings, filename: str, *, workspace_root: Path | None = None
) -> Path:
    root = resolve_workspace_dir(
        settings.data_dir, _workspace_root(workspace_root), env_name="MING_DATA_DIR"
    )
    return root / filename


def _output_path(
    settings: Settings, *parts: str, workspace_root: Path | None = None
) -> Path:
    return resolve_output_root(
        settings.output_dir, _workspace_root(workspace_root)
    ).joinpath(*parts)


def build_runtime(
    settings: Settings, *, workspace_root: Path | None = None
) -> tuple[CodexIndex, TelemetryStore, Orchestrator]:
    root = workspace_root
    codex = CodexIndex(_data_path(settings, "codex.db", workspace_root=root), TOOLS_ROOT)
    telemetry = TelemetryStore(_output_path(settings, "logs", "telemetry.db", workspace_root=root))
    iterations = IterationStore(_output_path(settings, "iterations", workspace_root=root))
    return codex, telemetry, Orchestrator(
        settings, codex, telemetry, iterations=iterations
    )


def build_image_artist(
    settings: Settings, *, workspace_root: Path | None = None
) -> tuple[TelemetryStore, IterationStore, ImageArtist]:
    root = _workspace_root(workspace_root)
    telemetry = TelemetryStore(_output_path(settings, "logs", "telemetry.db", workspace_root=root))
    iterations = IterationStore(_output_path(settings, "iterations", workspace_root=root))
    return telemetry, iterations, ImageArtist(settings, workspace_root=root)


def _print_report(report: object, as_json: bool) -> None:
    data = report.to_dict()  # type: ignore[attr-defined]
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    observed = data["observed"]
    estimate = data["estimate"]
    print(f"RUN: {data['run_id']}")
    print(f"MODE: {data['mode']}")
    print(f"STATUS: {data['status']}")
    print(f"ITERATION: {data['iteration_path']}")
    print(f"SPECIALISTS: {len(data['results'])}")
    print(
        f"VALIDATION: {observed['validation_passed']}/{observed['validation_total']} passed"
    )
    print(f"OBSERVED TOKENS: {observed['total_tokens']}")
    models = observed["actual_models"] or ["offline/deterministic"]
    print(f"ACTUAL MODELS: {', '.join(models)}")
    if observed.get("fallbacks"):
        print(f"FALLBACKS: {'; '.join(observed['fallbacks'])}")
    if estimate["hours"] is not None:
        print(
            f"ESTIMATED COMPARISON VALUE: ${estimate['comparison_value']:.2f} "
            f"({estimate['hours']} operator-supplied hours; not measured savings)"
        )
    print("\nDELIVERABLES")
    for result in data["results"]:
        print(f"\n[{result['definition']['role_name']}]\n{result['content']}")
    print("\nNEXT: Human review is required before using model output as implementation evidence.")


def command_doctor(settings: Settings, as_json: bool) -> int:
    codex, telemetry, _ = build_runtime(settings)
    telemetry.initialize()
    checks = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_supported": sys.version_info >= (3, 12),
        "sqlite": sqlite3.sqlite_version,
        "fts5": codex.fts5_available(),
        "codex_cards": codex.count(),
        "api_key_configured": bool(settings.api_key),
        "free_only": True,
        "image_model": settings.image_models[0],
        "image_estimated_cost_usd": settings.image_estimated_cost_usd,
        "image_model_estimates": dict(settings.image_model_estimates),
        "image_paid_opt_in_required": not settings.image_models[0].endswith(":free"),
        "base_url": settings.base_url,
        "data_dir": str(_data_path(settings, "").resolve()),
        "outputs_dir": str(_output_path(settings)),
        "logs_dir": str(_output_path(settings, "logs")),
        "iterations_dir": str(_output_path(settings, "iterations")),
        "images_dir": str(_output_path(settings, "images")),
    }
    checks["ready_offline"] = bool(checks["python_supported"] and checks["fts5"])
    checks["live_config_present"] = bool(
        checks["ready_offline"] and checks["api_key_configured"]
    )
    checks["live_provider_verified"] = False
    if as_json:
        print(json.dumps(checks, indent=2, sort_keys=True))
    else:
        for key, value in checks.items():
            print(f"{key}: {value}")
        if not checks["api_key_configured"]:
            print("live recovery: set OPENROUTER_API_KEY in .env; offline mode is available now")
    return 0 if checks["ready_offline"] else 1


def command_index(settings: Settings, as_json: bool) -> int:
    codex, _, _ = build_runtime(settings)
    result = codex.rebuild()
    payload = {
        "indexed": result.indexed,
        "skipped": result.skipped,
        "errors": list(result.errors),
        "database": str(codex.db_path),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Indexed {result.indexed} capability cards; skipped {result.skipped}.")
        for error in result.errors:
            print(f"ERROR: {error}")
    return 1 if result.errors else 0


def command_search(settings: Settings, query: str, limit: int, as_json: bool) -> int:
    codex, _, _ = build_runtime(settings)
    cards = codex.search(query, limit=limit)
    payload = [
        {
            "id": card.card_id,
            "title": card.title,
            "summary": card.summary,
            "tags": list(card.tags),
            "source": card.source,
            "score": card.score,
        }
        for card in cards
    ]
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not cards:
        print("No matching capability cards.")
    else:
        for card in payload:
            print(f"{card['id']} — {card['title']}\n  {card['summary']}\n  {card['source']}")
    return 0


async def command_run(settings: Settings, args: argparse.Namespace) -> int:
    _, _, orchestrator = build_runtime(settings)
    report = await orchestrator.run(
        args.brief,
        offline=args.offline,
        success_criteria=tuple(args.success),
        output_format="json" if args.output_json else "text",
        estimated_hours=args.estimated_hours,
    )
    if (
        report.status in {"completed", "completed_offline", "partial"}
        and report.iteration_path is None
    ):
        print(
            "WARNING: run finished, but the iteration snapshot could not be written.",
            file=sys.stderr,
        )
    _print_report(report, args.json)
    return 0 if report.status in {"completed", "completed_offline"} else 2


async def command_image(settings: Settings, args: argparse.Namespace) -> int:
    telemetry, iterations, artist = build_image_artist(settings)
    request = ImageRequest(
        need=args.need,
        purpose=args.purpose,
        audience=args.audience,
        style=args.style,
        avoid=args.avoid,
        exact_text=args.exact_text,
        aspect_ratio=args.aspect_ratio,
        resolution="1K",
        model=args.model,
    )
    if args.dry_run:
        payload = artist.preview(request)
        iteration_id = f"image-preview-{uuid.uuid4().hex[:12]}"
        iteration_path = iterations.path_for(iteration_id)
        payload.update(
            {
                "iteration_id": iteration_id,
                "iteration_kind": "image_preview",
                "iteration_path": str(iteration_path),
                "created_at": utc_now(),
            }
        )
        iterations.write(iteration_id, payload)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("IMAGE ARTIST DRY RUN — no API request was made")
            print(f"MODEL: {payload['model']}")
            print(f"COST CLASS: {payload['cost_class']}")
            print(f"ESTIMATED COST: ${payload['estimated_cost_usd']:.4f}")
            print(f"FRAME: {payload['aspect_ratio']} at {payload['resolution']}")
            print(f"ITERATION: {payload['iteration_path']}")
            print(f"\nPROMPT\n{payload['prompt']}")
        return 0

    try:
        artifact = await artist.generate(
            request,
            allow_paid=args.allow_paid,
            max_cost_usd=args.max_cost_usd,
            output=args.output,
            overwrite=args.overwrite,
        )
    except ImageGenerationCancelled as exc:
        if exc.artifact is not None:
            print(
                "WARNING: generation was cancelled after the provider response arrived; "
                f"preserved {exc.artifact.output_path}",
                file=sys.stderr,
            )
        raise
    telemetry_failed = False
    iteration_path: Path | None = None
    try:
        telemetry.record_image(artifact)
    except Exception as exc:
        telemetry_failed = True
        print(f"WARNING: failed to record image telemetry: {exc}", file=sys.stderr)

    payload = artifact.to_dict()
    iteration_id = f"image-{artifact.generation_id}"
    try:
        iteration_path = iterations.write(
            iteration_id,
            {
                "iteration_id": iteration_id,
                "iteration_kind": "image_generation",
                "artifact": payload,
            },
        )
        payload["iteration_path"] = str(iteration_path)
    except Exception as exc:
        payload["iteration_path"] = "unpersisted"
        print(f"WARNING: failed to record image iteration: {exc}", file=sys.stderr)

    if args.json:
        payload["telemetry_recorded"] = not telemetry_failed
        payload["iteration_recorded"] = iteration_path is not None
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        observed = (
            f"${artifact.observed_cost_usd:.6f}"
            if artifact.observed_cost_usd is not None
            else "not returned by provider"
        )
        print(f"IMAGE: {artifact.output_path}")
        print(f"MANIFEST: {artifact.manifest_path}")
        print(f"ITERATION: {iteration_path or 'unpersisted (audit write failed)'}")
        print(f"MODEL: {artifact.actual_model}")
        print(f"OBSERVED COST: {observed}")
        print(f"SHA256: {artifact.sha256}")
        print("REVIEW: Human visual review required before use.")
        if artifact.budget_exceeded:
            print(
                "WARNING: observed cost exceeded the authorized maximum; the billed artifact "
                "was preserved with provenance"
            )
        if telemetry_failed:
            print(
                "WARNING: image was generated and preserved, but telemetry database "
                "recording failed."
            )
        if iteration_path is None:
            print(
                "WARNING: image was generated and preserved, but iteration snapshot "
                "write failed."
            )
    if telemetry_failed or iteration_path is None:
        return 3
    return 2 if artifact.budget_exceeded else 0


def command_models(settings: Settings, as_json: bool) -> int:
    payload = settings.public_status()
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Provider: OpenRouter")
        print("Text cost route: explicitly free-only")
        print(f"API key configured: {payload['api_key_configured']}")
        for role, models in payload["models"].items():  # type: ignore[union-attr]
            print(f"{role}: {' -> '.join(models)}")
        image = payload["image_generation"]
        print(
            f"image: {' -> '.join(image['models'])} "  # type: ignore[index]
            f"(estimate ${image['estimated_cost_usd']:.4f}; explicit paid opt-in required)"  # type: ignore[index]
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ming",
        description="Local-first JIT specialist harness with free-only OpenRouter routing.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check local and live readiness")
    index = subparsers.add_parser("index", help="rebuild the capability codex")
    models = subparsers.add_parser("models", help="show safe model routing configuration")

    search = subparsers.add_parser("search", help="search capability cards")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)

    run = subparsers.add_parser("run", help="execute a bounded specialist run")
    run.add_argument("brief")
    run.add_argument("--offline", action="store_true", help="run deterministic local proof")
    run.add_argument("--success", action="append", default=[], help="observable criterion")
    run.add_argument("--output-json", action="store_true", help="require specialist JSON output")
    run.add_argument("--estimated-hours", type=_finite_non_negative_float)

    image = subparsers.add_parser(
        "image",
        help="plan or generate one cost-authorized image for the current need",
    )
    image.add_argument("need", help="the communication need the image must solve")
    image.add_argument("--purpose", default="general visual support")
    image.add_argument("--audience", default="the intended audience")
    image.add_argument(
        "--style", default="clear, intentional, context-appropriate visual design"
    )
    image.add_argument(
        "--avoid",
        default="generic stock imagery, clutter, illegible details, accidental text",
    )
    image.add_argument("--exact-text", help="the only text permitted in the image")
    image.add_argument(
        "--aspect-ratio", choices=SUPPORTED_ASPECT_RATIOS, default="1:1"
    )
    image.add_argument("--model", help="must be listed in MING_IMAGE_MODELS")
    image.add_argument(
        "--output",
        type=Path,
        help="filename or nested path beneath outputs/images",
    )
    image.add_argument("--dry-run", action="store_true", help="show the art brief without API use")
    image.add_argument(
        "--allow-paid", action="store_true", help="authorize one paid image request"
    )
    image.add_argument(
        "--max-cost-usd",
        type=_finite_non_negative_float,
        help="maximum authorized cost for this image",
    )
    image.add_argument(
        "--overwrite", action="store_true", help="replace an existing output and manifest"
    )

    evaluate = subparsers.add_parser("eval", help="run deterministic offline evaluation fixtures")
    for command_parser in (doctor, index, models, search, run, image, evaluate):
        command_parser.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = _settings()
        if args.command == "doctor":
            return command_doctor(settings, args.json)
        if args.command == "index":
            return command_index(settings, args.json)
        if args.command == "models":
            return command_models(settings, args.json)
        if args.command == "search":
            return command_search(settings, args.query, args.limit, args.json)
        if args.command == "run":
            return asyncio.run(command_run(settings, args))
        if args.command == "image":
            return asyncio.run(command_image(settings, args))
        if args.command == "eval":
            from evals.runner import run_evals

            return asyncio.run(run_evals(settings, as_json=args.json))
    except (ConfigurationError, GatewayError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"ming: {exc}", file=sys.stderr)
        return 2
    except asyncio.CancelledError:
        print("ming: cancelled", file=sys.stderr)
        return 130
    return 1
