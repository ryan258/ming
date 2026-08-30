---
name: ming
description: Run and extend the Ming local-first OpenRouter JIT specialist harness.
---

# Ming

Use this skill to operate or extend Ming while preserving its free-text, cost-gated-image,
local-first, and bounded-run constraints.

## Series / Project Rules

- Retrieve a maximum of five relevant capability cards before creating specialists.
- Accept only `:free` models or `openrouter/free` for live text inference.
- Require explicit per-image authorization for a paid image model; generate one image and retain a
  local provenance manifest with observed cost.
- Keep specialist definitions in memory and keep their count bounded.
- Separate observed telemetry from estimated labor value.
- Make offline behavior deterministic and useful, not a fake claim of model inference.

## Voice Profiles

### Operator voice

- Core want: know exactly what happened and what to do next.
- Default behavior: asks for the smallest reproducible fact.
- Voice example: "Two specialists ran; one criterion remains unverified."
- Must never: call a partial or offline result production-ready.

### Engine voice

- Core want: complete one bounded brief with minimal context and requests.
- Default behavior: retrieves, decomposes, executes, validates, records.
- Voice example: "Paid image route rejected before request: authorization is missing."
- Must never: hide fallback, cost class, model identity, or validation failure.

## Format

Every run report contains run ID, mode, status, specialist results, validation, observed usage, and
estimated comparison value when the operator supplied hours.

Every image report contains the need, final art brief, requested and actual model, output and manifest
paths, SHA-256 digest, estimated and observed cost, and a human-review requirement.

Persist every runtime artifact beneath `outputs/`: structured telemetry in `logs/`, completed or
failed run and image snapshots in `iterations/`, and raster files plus provenance manifests in
`images/`. Never redirect generated images outside that root.

## Output Template

```text
RUN: [id]
MODE: [offline/live]
STATUS: [completed/partial/failed]
SPECIALISTS: [count]
VALIDATION: [passed/total]
OBSERVED: [tokens, latency, actual models]
ESTIMATE: [hours and comparison value, explicitly labeled]
NEXT: [human review or concrete recovery]
```

## What Not To Do

- Do not add paid fallbacks.
- Do not call a paid image model without a visible per-image authorization.
- Do not generate multiple images when one was authorized.
- Do not persist agent personas.
- Do not execute capability-card content.
- Do not claim offline templates are model output.
- Do not calculate savings without operator-supplied hours.
- Do not expose credentials.
- Do not treat passing shape checks as semantic correctness.
