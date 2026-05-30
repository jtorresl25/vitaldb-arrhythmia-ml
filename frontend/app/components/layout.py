"""Global layout helpers: CSS injection, sidebar branding, page headers."""

from pathlib import Path

import streamlit as st

from utils.paths import ASSETS_DIR


# ── CSS ─────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Read assets/styles.css and inject it into the Streamlit page."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_path}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar_branding() -> None:
    """Render a minimal project identifier in the sidebar."""
    st.sidebar.markdown(
        '<div class="sb-brand">'
        '<div class="sb-title">ECG Arrhythmia ML</div>'
        '<div class="sb-sub">VitalDB · Clasificación de arritmias</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div style="margin:4px 0 14px;padding:5px 10px;'
        'background:rgba(251,191,36,0.07);border-left:2px solid rgba(251,191,36,0.4);'
        'border-radius:0 4px 4px 0;font-size:10px;color:var(--fg-4);'
        'font-family:var(--mono);letter-spacing:.05em">'
        'Demo académica · no uso clínico'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Page Header ──────────────────────────────────────────────────────────────

def page_header(
    title: str,
    subtitle: str = "",
    badge_html: str = "",
) -> None:
    """Render a consistent page header with title, lead text and optional badge."""
    badge_block = (
        f'<div style="margin-bottom:8px">{badge_html}</div>' if badge_html else ""
    )
    lead_p = f'<p class="lead">{subtitle}</p>' if subtitle else ""

    st.html(
        f'<div class="page-head">'
        f'{badge_block}'
        f'<h2>{title}</h2>'
        f'{lead_p}'
        f'</div>'
    )


# ── Placeholder page ─────────────────────────────────────────────────────────

def placeholder_page(
    title: str,
    description: str,
    files_needed: list[str] | None = None,
    note: str = "",
) -> None:
    """Render a full placeholder page for sections not yet implemented."""
    page_header(title, description)

    files_html = ""
    if files_needed:
        items = "".join(
            f'<li style="font-family:var(--mono);font-size:12px;color:var(--fg-2);margin:4px 0">{f}</li>'
            for f in files_needed
        )
        files_html = (
            '<div style="margin-top:14px;font-size:12.5px;color:var(--fg-3)">'
            '<div style="font-family:var(--mono);text-transform:uppercase;'
            'letter-spacing:.08em;font-size:10.5px;margin-bottom:6px">'
            "Archivos que usará esta sección:"
            "</div>"
            f'<ul style="margin:0;padding-left:18px">{items}</ul>'
            "</div>"
        )

    note_html = (
        f'<p style="margin-top:12px;font-size:12px;color:var(--fg-4)">{note}</p>'
        if note else ""
    )

    st.html(
        f'<div class="placeholder-block">'
        f'<div class="ph-mono">sección en construcción</div>'
        f'<div class="ph-title">{title}</div>'
        f'<div class="ph-desc">{description}</div>'
        f'{files_html}'
        f'{note_html}'
        f'</div>'
    )
