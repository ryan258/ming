# Architecture and Trust Boundaries

## Component Layout & Module Map

Ming is organized into bounded subsystems with clear separation of responsibilities:

- **`engine/`**: Core runtime, orchestration, and boundary enforcement.
  - `config.py`: Environment parsing, workspace boundary checks, model pricing, and free-only route enforcement.
  - `contracts.py`: Strongly-typed dataclasses for cards, specialists, tasks, events, validations, and telemetry records.
  - `bus.py`: Async in-memory fan-out event bus with queue saturation safeguards.
  - `validation.py`: Deterministic shape, JSON syntax, required key, and substring checks.
  - `llm.py`: OpenRouter client enforcing free text routes, model preference fallback chains, and sanitized error handling.
  - `telemetry.py`: Local SQLite store (`outputs/logs/telemetry.db`) tracking runs, events, calls, and image costs.
  - `jit_spawner.py`: Ephemeral specialist spawner creating bounded in-memory agents.
  - `image_generator.py`: Communication-need art brief composer, cost authorizer, and raster artifact generator with JSON provenance sidecars.
  - `orchestrator.py`: Solo orchestrator coordinating retrieval, planning, execution, validation, and iteration snapshots.
  - `output_store.py`: Workspace-confined atomic writer for run and image iteration JSONs.
- **`codex/`**: Capability indexer and curated Markdown cards.
  - `indexer.py`: SQLite FTS5 indexer with punctuation sanitization and BM25 ranking.
  - `tools/`: Modular Markdown capability cards across creative, data, devops, security, and web domains.
- **`ming/`**: Primary CLI interface and package entrypoints.
  - `cli.py`: Subcommand implementations (`doctor`, `index`, `search`, `run`, `image`, `models`, `eval`).
- **`ui/`**: Optional operations console.
  - `app.py`: Streamlit interface providing codex browsing, orchestrator runs, image workbench, and telemetry auditing.
- **`evals/`**: Deterministic offline fixtures.
  - `runner.py`: Offline evaluation runner checking capability retrieval and validation criteria.
  - `test_cases.json`: Grounded regression cases for offline validation.

## Runtime Flow

1. The CLI/UI loads configuration. Process environment wins over `.env`.
2. The orchestrator searches the FTS5 codex using terms from the brief.
3. A planner produces one to three task specifications. Offline mode uses a deterministic planner.
4. The JIT spawner creates in-memory specialist definitions with selected capability cards.
5. Specialists execute sequentially by default to respect free-tier constraints.
6. Validators grade explicit success criteria and output-shape requirements.
7. Telemetry stores run/call facts locally. The final report names the actual response model.

## Image-Artist Flow

1. The operator supplies a need, use, audience, art direction, exclusions, and aspect ratio.
2. Ming deterministically composes a bounded art brief. `--dry-run` stops here without a key or API.
3. Free image models need no spending authorization. A paid image model requires `--allow-paid` and
   `--max-cost-usd` at or above the configured estimate.
4. Ming requests exactly one 1K image through `POST /api/v1/images`. Cross-model fallback is disabled;
   OpenRouter may fail over among providers serving the selected model.
5. The base64 response is size-limited, decoded, and checked against its claimed raster media type.
6. The image and JSON provenance manifest are written as a pair inside `outputs/images/`; if
   either write fails, a fresh run is rolled back and `--overwrite` restores the previous files.
7. Image telemetry records the actual model, observed cost, digest, latency, and output path.

## Dual Interface Architecture

- **CLI (`ming/cli.py`)**: Designed for scriptability, continuous integration, and rapid operator interaction. Outputs can be human-readable formatted text or structured JSON (`--json`).
- **Streamlit Dashboard (`ui/app.py`)**: Optional visual operations console reading directly from the same engine, codex, and telemetry store without duplicating data or business logic.

## Security Boundaries

- The official `https://openrouter.ai/api/v1` origin is pinned by default. Custom origins require the
  explicit `MING_ALLOW_CUSTOM_BASE_URL=true` opt-in.
- API keys are never included in status output, telemetry, events, or exceptions produced by Ming.
- Runtime writes share one workspace-confined `outputs/` root: SQLite telemetry under `logs/`,
  durable run and image-attempt snapshots under `iterations/`, and raster artifacts plus provenance
  under `images/`. The capability index is separately rebuildable under workspace-confined `data/`.
- Text-model routes are free-only by construction. Text fallback cannot cross into a paid model.
- Image generation is a distinct spending boundary. Paid image calls require two explicit CLI/UI
  authorizations, request one image only, and compare the configured price estimate before transport.
- Image output cannot escape the workspace, silently overwrite an existing file, exceed the size
  limit, or masquerade as a different raster type.
- Capability cards are inert context. They are not shell commands and are never executed by Ming.
- SQLite uses parameterized statements. The indexer only ingests `.md` files beneath its configured
  tools root.
- The application does not stage, commit, push, deploy, publish, or call arbitrary local tools.

## Prototype Limitations

The planner and validators constrain common output failures; they do not establish correctness or
safety. Model output requires human review. Free model availability and quality are outside Ming's
control. Token ROI is a comparison estimate, not realized financial value.

The OpenRouter Images API does not expose a request-level maximum-cost parameter. Ming's
`--max-cost-usd` is therefore a local authorization threshold checked against the configured model
estimate before transport; the exact cost arrives in response usage after a completed generation.
The default model is pinned to a flat per-image price, but model pricing remains external and can
change. Refresh `MING_IMAGE_ESTIMATED_COST_USD` (or `MING_IMAGE_MODEL_COSTS` for multiple paid models)
whenever the configured image model or price changes. Image request fields are taken from a pinned
capability table; unknown models are rejected before authorization.
