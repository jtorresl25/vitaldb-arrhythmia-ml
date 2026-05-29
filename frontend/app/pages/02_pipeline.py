"""Página 02 — Pipeline metodológico."""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from components.badges import badge, badge_row
from components.cards import callout
from components.layout import page_header

# ── Rutas a los HTML ──────────────────────────────────────────────────────────

_METODOLOGIA_DIR = (
    Path(__file__).parent.parent.parent / "streamlit_reference" / "Metodologia"
)
_HTML_SIMPLE = _METODOLOGIA_DIR / "Pipeline del proyecto.html"
_HTML_DETAIL = _METODOLOGIA_DIR / "pipeline_metodologico.html"

# ── 1 · Header ────────────────────────────────────────────────────────────────

page_header(
    "Pipeline metodológico",
    "De señales ECG intraoperatorias a evaluación multiclase de ritmos cardíacos.",
    badge_html=badge("Metodología", "info"),
)

st.markdown(
    badge_row(
        badge("VitalDB", "info"),
        badge("ECG", "info"),
        badge("Machine Learning", "info"),
        badge("Demo académica · No uso clínico", "warn"),
    ),
    unsafe_allow_html=True,
)

# ── 2 · Mapa visual simplificado ─────────────────────────────────────────────

components.html(
    _HTML_SIMPLE.read_text(encoding="utf-8"),
    height=1400,
    scrolling=False,
)

# ── 3 · Callout ───────────────────────────────────────────────────────────────

callout(
    "info",
    "Resumen del flujo",
    "Este mapa resume el flujo general desde el problema y los datos VitalDB "
    "hasta el modelado, evaluación y despliegue en Streamlit.",
)

# ── 4 · Metodología detallada ─────────────────────────────────────────────────

with st.expander("Ver metodología detallada"):
    components.html(
        _HTML_DETAIL.read_text(encoding="utf-8"),
        height=2600,
        scrolling=True,
    )
