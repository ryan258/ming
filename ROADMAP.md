# Ming Roadmap

Ming is a local-first agent harness that turns one brief into a small, observable run: a solo
orchestrator searches a curated capability codex, creates only the specialists it needs, routes
their inference through free OpenRouter models, validates their outputs, and records a local audit
trail. A separate image artist turns a current communication need into one explicitly
cost-authorized raster image with local provenance.

This roadmap outlines completed milestones, upcoming phases, and persistent architectural non-goals.

---

## Milestone Progress

### Phase 1: Core Engine & Interfaces (Completed)

- [x] **Capability Codex (SQLite FTS5)**
  - Markdown capability card parsing (`id`, `title`, `description`, `triggers`, `inputs`, `outputs`, `tags`).
  - Parameterized SQLite FTS5 indexer (`codex/indexer.py`) with sanitization of FTS syntax punctuation.
  - Initial seed capability cards (`creative`, `data`, `devops`, `security`, `web`).
- [x] **Ephemeral JIT Specialist Engine**
  - Contract dataclasses for cards, specialists, tasks, events, validations, and telemetry (`engine/contracts.py`).
  - Asynchronous fan-out event bus (`engine/bus.py`) with bounded subscriber queues.
  - JIT spawner (`engine/jit_spawner.py`) creating bounded, in-memory specialists.
  - Deterministic validators (`engine/validation.py`) checking output shapes and criteria.
- [x] **Strict Free-Only Text Inference Gateway**
  - OpenRouter client (`engine/llm.py`) enforcing free model naming (`:free` suffix or `openrouter/free`).
  - Rejection of paid models before any network transport.
  - Configurable fallback chains per role ending deterministically at `openrouter/free`.
- [x] **Cost-Authorized Image Artist**
  - Art brief composer (`engine/image_generator.py`) translating communication needs into bounded prompts.
  - Dry-run mode saving iteration plans under `outputs/iterations/` without API requests or keys.
  - Two-level spending authorization: `--allow-paid` flag and per-request `--max-cost-usd` threshold check.
  - Bounded 1K single-image generation, raster header verification, size capping, and atomic write with JSON provenance sidecars under `outputs/images/`.
- [x] **Dual Interfaces & Tooling**
  - Command-line interface (`ming/cli.py`) with subcommands: `doctor`, `index`, `search`, `run`, `image`, `models`, `eval`.
  - Operations dashboard in Streamlit (`ui/app.py`) for codex inspection, live/offline execution, image generation workbench, and telemetry exploration.
  - Deterministic eval runner (`evals/runner.py`) executing offline regression fixtures.
  - Portable launcher (`run.sh`) with automatic Python 3.12 detection via `uv`.
- [x] **Deterministic Test Coverage**
  - 98 unit tests covering config, contracts, bus, validation, indexer, llm, orchestrator, output store, telemetry, and image generator.

---

### Phase 2: Codex Expansion & Card Authoring (Next)

- [ ] **Capability Card Linter and Validator**
  - CLI subcommand (`ming lint-codex`) to validate card metadata, verify schema adherence, and detect broken links or redundant tags before indexing.
  - Automated checks for duplicate IDs, missing required fields, and excessive context lengths.
- [ ] **Expanded Seed Capability Library**
  - Additional operational cards for common developer workflows (git diff review, shell command synthesis, database schema design, documentation extraction).
  - Domain-specific validation criteria patterns.
- [ ] **Structured Telemetry & Audit Export**
  - Export utilities to serialize runs, events, and image provenance from `outputs/logs/telemetry.db` into portable NDJSON or CSV summaries.
  - CLI subcommand for run audit summaries and cost tracking aggregations.

---

### Phase 3: Observability & Streaming

- [ ] **Terminal Event Streaming**
  - Optional real-time console rendering of event bus transitions and specialist progress during long runs.
  - Rich progress bars and structured log levels in terminal output.
- [ ] **Streamlit Live Monitoring**
  - Reactive event updates via Streamlit server components or polling session state.
  - Interactive drill-down into token metrics, prompt construction, and specialist payload history.
- [ ] **Model Health & Availability Probing**
  - Optional non-blocking health check command to probe whether free-tier OpenRouter models are currently responsive before executing long tasks.

---

### Phase 4: Golden Evaluation & Benchmarking

- [ ] **Expanded Offline Golden Suites**
  - Multi-case evaluation matrix covering complex briefs, ambiguous requests, and boundary failures.
  - Precision and recall benchmarks for the SQLite FTS5 capability retriever.
- [ ] **Mock Provider Replay Harness**
  - Fixture replay system allowing full pipeline integration testing with captured provider responses.
  - Resilience verification for handling transient rate limits and provider failovers.

---

## Architectural Non-Goals & Boundaries

Ming maintains strict operational boundaries to remain a trustworthy personal tool:

1. **No Silent Spending**: Text routes are structurally constrained to free endpoints. Paid image generation requires explicit CLI/UI authorization flags every time.
2. **No Persistent Agent Personas**: Specialists exist only in memory during the execution of a bounded task.
3. **No Unprompted External Actions**: Ming does not stage, commit, push, deploy, or publish git repositories, code, or artifacts.
4. **No Secret Persistence**: API keys and sensitive tokens are never stored in telemetry databases, iteration snapshots, or image manifests.
5. **No Speculative Scaling**: Ming is single-user, workspace-confined, and designed for local execution without distributed worker infrastructure or cloud dependencies.
