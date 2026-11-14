# Ming

Ming is a local-first agent harness that turns one brief into a small, observable run: a solo
orchestrator searches a curated capability codex, creates only the specialists it needs, routes
their inference through free OpenRouter models, validates their outputs, and records a local audit
trail. A separate image artist turns a current communication need into one explicitly
cost-authorized raster image with local provenance. It is an honest low-volume prototype, not a
claim of unlimited autonomous labor.

The useful part works without an API key or third-party package. Offline mode exercises the event
bus, SQLite FTS5 codex, specialist lifecycle, validation, and telemetry deterministically. A live
key enables OpenRouter inference. The optional Streamlit interface visualizes the same engine.

## Quick start

Python 3.12 or newer is required. If the system `python3` is older but `uv` is installed, `run.sh`
selects an installed 3.12 runtime automatically.

```sh
./run.sh doctor
./run.sh index
./run.sh search "inspect a Python file for security issues"
./run.sh run "Inspect a Python web handler and return a concise security checklist" --offline
```

For a live run:

```sh
cp .env.example .env
# Add OPENROUTER_API_KEY to .env, or export it in the shell.
./run.sh run "Design a small, tested JSON API" --success "mentions tests"
```

Ming loads `.env` itself without overriding values already present in the process environment.
Every configured live **text** model must be `openrouter/free` or end in `:free`; a paid text model
is rejected before any network request.

### Image artist

OpenRouter currently lists no `:free` image-output route. Ming therefore defaults to
`krea/krea-2-medium-turbo`, a 1K model listed at approximately $0.015 per image as of August 29,
2026. A dry run plans the image without a key, API request, or charge and saves the prepared brief
as an iteration record under `outputs/iterations/`:

```sh
./run.sh image "A calm hero image for a local-first operations dashboard" \
  --purpose "website hero" \
  --audience "technical operators" \
  --style "dark editorial technology illustration with quiet green accents" \
  --avoid "robots, glowing brains, stock photography, text, visual clutter" \
  --aspect-ratio 16:9 \
  --dry-run
```

A real generation requires both an explicit paid opt-in and a per-request authorization:

```sh
./run.sh image "A calm hero image for a local-first operations dashboard" \
  --purpose "website hero" \
  --style "dark editorial technology illustration with quiet green accents" \
  --aspect-ratio 16:9 \
  --allow-paid \
  --max-cost-usd 0.02
```

Ming requests one 1K image, saves it beneath `outputs/images/`, and writes a JSON sidecar with the
full art brief, actual model, media type, digest, usage, and observed cost. The default output model
has a flat listed price. `--max-cost-usd` is a local authorization check against the estimate bound
to the selected model, not a provider-side hard spending cap. A single paid model uses
`MING_IMAGE_ESTIMATED_COST_USD`; additional paid models require `MING_IMAGE_MODEL_COSTS`. Always
refresh those estimates when changing image models. Ming only sends `resolution` and `aspect_ratio`
when the selected model is in its pinned capability table.

For the optional UI:

```sh
uv sync --extra ui
./run.sh --ui
```

## Commands

- `doctor` — report Python, FTS5, codex, telemetry, key presence, and free-only routing state.
  `live_config_present` means a key is configured; it does not authenticate the provider.
- `index` — rebuild the local FTS5 codex from Markdown capability cards.
- `search QUERY` — show the minimal capability context a specialist would receive.
- `run BRIEF` — execute the harness; add `--offline` for deterministic local proof.
- `image NEED` — preview an art brief or generate one cost-authorized workspace-confined image.
- `models` — show the configured preference/fallback chains without exposing credentials.
- `eval` — run deterministic fixtures through the offline engine.

All runtime artifacts share one gitignored `outputs/` root:

```text
outputs/
├── logs/telemetry.db       # structured runs, events, calls, and image cost telemetry
├── iterations/*.json      # every completed or failed run and every image preview/generation
└── images/*                # generated raster files and adjacent provenance manifests
```

The FTS5 capability index remains at `data/codex.db` because it is rebuildable application state,
not a runtime output. Iteration records and image manifests store prompts locally, so do not submit
secrets in a brief. Set `MING_OUTPUT_DIR` to rename the root, but it must remain inside the current
workspace. A relative `image --output` value is resolved beneath `outputs/images/`; images cannot be
redirected elsewhere. No command stages, commits, pushes, deploys, or publishes anything.

## Architecture

```text
CLI / Streamlit
      |
      +--- solo orchestrator --- free-only OpenRouter text gateway
      |          |
      |      async event bus --- JIT specialists (memory only)
      |
      +--- image artist --- cost authorization --- OpenRouter Images API
      |          |
      |      outputs/images raster + provenance manifest
      |
outputs/iterations JSON + outputs/logs telemetry
      |
SQLite FTS5 capability codex + deterministic validation
```

The named free model roster is configuration, not a promise: free endpoints rotate. Ming attempts
the role-specific preference chain and always ends at `openrouter/free`. Rate limiting is local and
conservative; OpenRouter still enforces account-level limits.

See [docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md) for scope,
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation boundaries, and
[ROADMAP.md](ROADMAP.md) for milestone progress. The saved
[first output](examples/first-output.json) is the JSON deliverable from a verified offline run; its
labels make clear that it proves orchestration plumbing rather than model inference.

## Verification

```sh
uv run pytest
# or
python3 -m unittest discover -s tests -v
python3 -m ming eval
python3 -m ming run "Create a deterministic JSON summary" --offline --json
python3 -m ming image "A dashboard hero" --aspect-ratio 16:9 --dry-run
```

Live text and image smoke tests are intentionally not automatic because they consume account
requests; an image request is also billed when it completes.
