# Ming Project Bible

## What Is It

Ming is a local-first Python CLI and optional Streamlit dashboard for decomposing a user brief into
a bounded set of ephemeral AI specialists. A solo orchestrator retrieves relevant capability cards,
routes inference through explicitly free OpenRouter endpoints, validates outputs, and records local
telemetry. Its image artist can also create one need-driven visual through a separate, explicitly
cost-authorized OpenRouter image boundary.

It is for developers and technical operators experimenting with auditable, low-cost agent workflows.
It is not positioned as a production SLA, a replacement for human review, or an unlimited swarm.

## Setting

Ming lives in the terminal first. Its visual language is a dark operations console: quiet, legible,
and evidence-oriented. The optional dashboard reads the same local codex and telemetry used by the
CLI rather than maintaining a second source of truth.

## Tone

Sober, capable, and transparent. Ming reports what ran, which model actually answered, what passed,
and what remains human-reviewed.

## Format Rules

Every run must have a brief, a run identifier, a bounded specialist count, a capability context,
events, validation results, and telemetry. Live text routes must remain free-only. Image generation
must remain single-output, locally persisted, provenance-backed, and explicitly authorized whenever
paid. Logs, run iterations, image previews, generated rasters, and image provenance must remain
durable beneath the workspace-confined `outputs/` root. The capability index is rebuildable state
and may remain under `data/`.
Specialists exist only in memory for their assigned task. Generated evidence remains local by
default.

## What It Is Not

1. It is not an autonomous production deployment system.
2. It is not a promise that any named free model will remain available.
3. It is not a dollar-savings claim disguised as measured human labor.

## Connection to the Universe

Ming stands alone as a constraint-driven orchestration proof. Its Markdown capability cards can
describe tools from other repositories, but Ming never assumes permission to execute or alter those
repositories.

## Production Window

A capability card should take 10–20 minutes to author. An offline run should finish in seconds. A
live run is intentionally limited to one planning request plus at most three specialist requests,
subject to free-tier availability and account limits. An image unit is one dry-run art brief or one
authorized 1K image plus its provenance manifest.
