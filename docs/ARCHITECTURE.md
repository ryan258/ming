# Architecture and Trust Boundaries

## Runtime flow

1. The CLI/UI loads configuration. Process environment wins over `.env`.
2. The orchestrator searches the FTS5 codex using terms from the brief.
3. A planner produces one to three task specifications. Offline mode uses a deterministic planner.
4. The JIT spawner creates in-memory specialist definitions with selected capability cards.
5. Specialists execute sequentially by default to respect free-tier constraints.
6. Validators grade explicit success criteria and output-shape requirements.
7. Telemetry stores run/call facts locally. The final report names the actual response model.

## Image-artist flow

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

## Security boundaries

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

## Prototype limitations

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
