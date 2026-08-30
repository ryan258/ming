from __future__ import annotations

import asyncio
import html
import json
import sys
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import ConfigurationError, Settings  # noqa: E402
from engine.contracts import utc_now  # noqa: E402
from engine.image_generator import ImageRequest  # noqa: E402
from ming.cli import build_image_artist, build_runtime  # noqa: E402

st.set_page_config(page_title="Ming", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #0a0e14; color: #d8e2ef; }
      [data-testid="stMetric"] { background:#111923; border:1px solid #243244;
        border-radius:12px; padding:14px; }
      .ming-kicker { color:#6ee7b7; letter-spacing:.16em; font-size:.75rem; font-weight:700; }
      .ming-card { background:#111923; border-left:3px solid #6ee7b7; padding:16px;
        border-radius:4px 12px 12px 4px; margin:.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    settings = Settings.from_env(PROJECT_ROOT / ".env")
except ConfigurationError as exc:
    st.error(str(exc))
    st.stop()

codex, telemetry, orchestrator = build_runtime(settings, workspace_root=PROJECT_ROOT)
_, image_iterations, image_artist = build_image_artist(
    settings, workspace_root=PROJECT_ROOT
)
if codex.count() == 0:
    codex.rebuild()

st.markdown('<div class="ming-kicker">SOVEREIGN AGENT HARNESS</div>', unsafe_allow_html=True)
st.title("Ming")
st.caption(
    "Bounded JIT specialists · local capability codex · free text routes · cost-gated image artist"
)

run_tab, image_tab, codex_tab, telemetry_tab, eval_tab = st.tabs(
    ["Live swarm run", "Image artist", "Codex explorer", "Telemetry", "Eval suite"]
)

with run_tab:
    left, right = st.columns([2, 1])
    with left:
        brief = st.text_area(
            "Brief",
            "Inspect a Python web handler and return a concise security checklist",
            height=140,
        )
        criteria_text = st.text_input("Success criteria (comma separated)", "mentions review")
    with right:
        offline = st.toggle("Offline deterministic proof", value=not bool(settings.api_key))
        output_format = st.selectbox("Output format", ["text", "json"])
        estimated_hours = st.number_input(
            "Human-hours comparison (optional)", min_value=0.0, value=0.0, step=0.25
        )
    if st.button("Run bounded swarm", type="primary", width="stretch"):
        if not brief.strip():
            st.warning("Enter a brief.")
        else:
            criteria = tuple(item.strip() for item in criteria_text.split(",") if item.strip())
            with st.status("Running…", expanded=True) as status:
                try:
                    report = asyncio.run(
                        orchestrator.run(
                            brief,
                            offline=offline,
                            success_criteria=criteria,
                            output_format=output_format,
                            estimated_hours=estimated_hours or None,
                        )
                    )
                except Exception as exc:  # Streamlit presents bounded recovery in the UI.
                    status.update(label="Run failed", state="error")
                    st.error(str(exc))
                else:
                    status.update(label=f"Run {report.status}", state="complete")
                    data = report.to_dict()
                    metrics = st.columns(4)
                    metrics[0].metric("Specialists", len(report.results))
                    metrics[1].metric("Tokens", report.total_tokens)
                    passed, total = report.validation_counts
                    metrics[2].metric("Validation", f"{passed}/{total}")
                    metrics[3].metric("Mode", report.mode)
                    for result in report.results:
                        st.markdown(
                            '<div class="ming-card">'
                            f'<b>{html.escape(result.definition.role_name)}</b><br>'
                            f'Model: {html.escape(result.model)}</div>',
                            unsafe_allow_html=True,
                        )
                        language = "json" if output_format == "json" else None
                        st.code(result.content, language=language)
                    with st.expander("Full audit record"):
                        st.json(data)

with image_tab:
    st.caption(
        "Dry-run is free. Live generation uses the configured image model and requires explicit "
        "per-image spending authorization."
    )
    image_need = st.text_area(
        "What does the image need to do?",
        "Create a calm, legible hero image for a local-first agent operations dashboard",
        key="image_need",
    )
    image_left, image_right = st.columns(2)
    with image_left:
        image_purpose = st.text_input("Use", "dashboard hero image")
        image_audience = st.text_input("Audience", "technical operators")
        image_style = st.text_input(
            "Art direction", "dark editorial technology illustration with quiet green accents"
        )
        image_avoid = st.text_input(
            "Avoid", "robots, glowing brains, generic stock imagery, text, visual clutter"
        )
    with image_right:
        image_ratio = st.selectbox(
            "Aspect ratio", ["1:1", "16:9", "4:3", "3:2", "4:5", "2:3", "9:16"]
        )
        image_model = st.selectbox("Image model", list(settings.image_models))
        paid_authorized = st.checkbox("Authorize one paid image request")
        image_max_cost = st.number_input(
            "Maximum authorized cost (USD)",
            min_value=0.0,
            value=float(max(settings.image_estimated_cost_usd, 0.02)),
            step=0.005,
            format="%.3f",
        )
    image_request = ImageRequest(
        need=image_need,
        purpose=image_purpose,
        audience=image_audience,
        style=image_style,
        avoid=image_avoid,
        aspect_ratio=image_ratio,
        model=image_model,
    )
    preview_col, generate_col = st.columns(2)
    if preview_col.button("Preview art brief", width="stretch"):
        try:
            preview = image_artist.preview(image_request)
            iteration_id = f"image-preview-{uuid.uuid4().hex[:12]}"
            iteration_path = image_iterations.path_for(iteration_id)
            preview.update(
                {
                    "iteration_id": iteration_id,
                    "iteration_kind": "image_preview",
                    "iteration_path": str(iteration_path),
                    "created_at": utc_now(),
                }
            )
            image_iterations.write(iteration_id, preview)
            st.session_state["image_preview"] = preview
        except Exception as exc:
            st.error(str(exc))
    if generate_col.button("Generate one image", type="primary", width="stretch"):
        artifact = None
        try:
            artifact = asyncio.run(
                image_artist.generate(
                    image_request,
                    allow_paid=paid_authorized,
                    max_cost_usd=image_max_cost,
                )
            )
        except Exception as exc:
            st.error(str(exc))

        if artifact is not None:
            telemetry_failed = False
            iteration_path = None
            try:
                telemetry.record_image(artifact)
            except Exception as exc:
                telemetry_failed = True
                st.warning(f"Telemetry database write failed: {exc}")

            iteration_id = f"image-{artifact.generation_id}"
            try:
                iteration_path = image_iterations.write(
                    iteration_id,
                    {
                        "iteration_id": iteration_id,
                        "iteration_kind": "image_generation",
                        "artifact": artifact.to_dict(),
                    },
                )
            except Exception as exc:
                st.warning(f"Iteration snapshot write failed: {exc}")

            st.image(artifact.output_path, caption=f"Generated with {artifact.actual_model}")
            st.success(
                "Saved image and provenance manifest. "
                f"Cost: {artifact.observed_cost_usd}"
            )
            rendered = artifact.to_dict()
            rendered["iteration_path"] = str(iteration_path) if iteration_path else "unpersisted"
            rendered["telemetry_recorded"] = not telemetry_failed
            rendered["iteration_recorded"] = iteration_path is not None
            st.json(rendered)
    if "image_preview" in st.session_state:
        with st.expander("Prepared art brief", expanded=True):
            st.json(st.session_state["image_preview"])

with codex_tab:
    query = st.text_input("Search capability cards", "security python")
    if query:
        cards = codex.search(query)
        st.caption(f"Injecting {len(cards)} of {codex.count()} indexed cards")
        for card in cards:
            with st.expander(f"{card.title} · {card.card_id}"):
                st.write(card.summary)
                st.caption(f"Tags: {', '.join(card.tags)} · Source: {card.source}")
                st.markdown(card.body)

with telemetry_tab:
    summary = telemetry.summary()
    cols = st.columns(4)
    cols[0].metric("Recorded calls", summary["calls"])
    cols[1].metric("Observed tokens", summary["tokens"])
    cols[2].metric("Average latency", f"{summary['average_latency_ms']} ms")
    cols[3].metric(
        "Generated images",
        summary["images"],
        help=f"Observed image cost: ${summary['image_observed_cost_usd']:.6f}",
    )
    st.caption("Offline runs have zero model calls. Values are observed, not projected savings.")
    st.dataframe(summary["runs"], width="stretch")

with eval_tab:
    st.write(
        "Run the deterministic retrieval and output-shape fixtures without consuming API quota."
    )
    st.code("python3 -m ming eval")
    st.json(json.loads((PROJECT_ROOT / "evals" / "test_cases.json").read_text()))
