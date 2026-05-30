"""Página 06 — Matriz de confusión binaria normal/anormal."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.layout import page_header, page_footer
from components.cards  import callout, card_header, kv_table, metric_card, section_title
from components.badges import badge, badge_row
from utils.loaders import (
    load_confusion_matrix_csv,
    load_binary_metrics,
    load_model_metadata,
    confusion_matrix_figure_path,
)

# ── Constants ──────────────────────────────────────────────────────────────────
_LABELS = ["Normal", "Anormal"]

_LABEL_MAP = {
    "n": "Normal", "normal": "Normal", "0": "Normal",
    "abnormal": "Anormal", "anormal": "Anormal", "1": "Anormal",
    "no n": "Anormal", "not n": "Anormal",
}

def _norm(x: str) -> str:
    return _LABEL_MAP.get(str(x).strip().lower(), str(x).title())


# ── Bootstrap ──────────────────────────────────────────────────────────────────
meta   = load_model_metadata()
winner = meta.get("winner_model", "—").replace("_", " ").title() if meta else "—"

page_header(
    "Matriz de confusión",
    "Evaluación binaria del modelo final: Normal vs Anormal.",
    badge_html=badge_row(badge(winner, "winner"), badge("Binario", "ok")),
)

# ── Callout metodológico ───────────────────────────────────────────────────────
callout(
    "info",
    "Cómo leer esta matriz",
    "<b>Filas</b> = clase real &nbsp;·&nbsp; "
    "<b>Columnas</b> = clase predicha &nbsp;·&nbsp; "
    "<b>Diagonal</b> = predicciones correctas. "
    "En un problema binario: "
    "<b>TN</b> = Normal detectado como Normal &nbsp;·&nbsp; "
    "<b>TP</b> = Anormal detectado como Anormal &nbsp;·&nbsp; "
    "<b>FP</b> = Normal clasificado como Anormal (falsa alarma) &nbsp;·&nbsp; "
    "<b>FN</b> = Anormal clasificado como Normal (caso no detectado). "
    "Los <b>FN son especialmente críticos</b>: representan anormalidades no detectadas.",
)

st.write("")

# ── Load data ──────────────────────────────────────────────────────────────────
df_cm_csv  = load_confusion_matrix_csv()
df_bin_met = load_binary_metrics()
cm_png     = confusion_matrix_figure_path()


# ── Matrix parser ──────────────────────────────────────────────────────────────
def _parse_matrix(df: "pd.DataFrame | None") -> "np.ndarray | None":
    """Return 2×2 ndarray [Normal row, Anormal row] × [Normal col, Anormal col]."""
    if df is None:
        return None

    idx = {"Normal": 0, "Anormal": 1}
    mat = np.zeros((2, 2), dtype=float)

    # Long format: real_label / y_true / true_label  +  predicted_label / y_pred  +  count
    real_col = next(
        (c for c in df.columns if c.lower() in ("real_label", "y_true", "true_label")), None
    )
    pred_col = next(
        (c for c in df.columns if c.lower() in ("predicted_label", "y_pred", "pred_label")), None
    )
    cnt_col = next(
        (c for c in df.columns if c.lower() in ("count", "value", "n")), None
    )

    if real_col and pred_col and cnt_col:
        for _, row in df.iterrows():
            r = idx.get(_norm(str(row[real_col])))
            c = idx.get(_norm(str(row[pred_col])))
            if r is not None and c is not None:
                mat[r, c] += float(row[cnt_col])
        return mat if mat.sum() > 0 else None

    # Wide format: first column = row label (true_class), rest = predicted columns
    try:
        df2 = df.set_index(df.columns[0]).copy()
        valid_rows = [r for r in df2.index if _norm(str(r)) in ("Normal", "Anormal")]
        valid_cols = [c for c in df2.columns if _norm(str(c)) in ("Normal", "Anormal")]
        if not valid_rows or not valid_cols:
            return None
        df2 = df2.loc[valid_rows, valid_cols]
        for real_raw in df2.index:
            ri = idx.get(_norm(str(real_raw)))
            if ri is None:
                continue
            for pred_raw in df2.columns:
                ci = idx.get(_norm(str(pred_raw)))
                if ci is None:
                    continue
                mat[ri, ci] = float(df2.loc[real_raw, pred_raw])
        return mat if mat.sum() > 0 else None
    except Exception:
        pass

    return None


mat = _parse_matrix(df_cm_csv)

# ── Derive / load metrics ──────────────────────────────────────────────────────
def _safe_float(v) -> "float | None":
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None

if df_bin_met is not None and not df_bin_met.empty:
    _r = df_bin_met.iloc[0]
    tn   = int(_safe_float(_r.get("tn_normal",              0)) or 0)
    fp   = int(_safe_float(_r.get("fp_abnormal_false_alarm",0)) or 0)
    fn   = int(_safe_float(_r.get("fn_abnormal_missed",     0)) or 0)
    tp   = int(_safe_float(_r.get("tp_abnormal",            0)) or 0)
    acc  = _safe_float(_r.get("accuracy"))
    sens = _safe_float(_r.get("sensitivity_recall_abnormal"))
    spec = _safe_float(_r.get("specificity_recall_normal"))
    prec = _safe_float(_r.get("precision_abnormal"))
    f1   = _safe_float(_r.get("f1_abnormal"))
elif mat is not None:
    tn, fp, fn, tp = int(mat[0, 0]), int(mat[0, 1]), int(mat[1, 0]), int(mat[1, 1])
    total = tn + fp + fn + tp or 1
    acc   = (tn + tp) / total
    sens  = tp / (tp + fn) if (tp + fn) > 0 else None
    spec  = tn / (tn + fp) if (tn + fp) > 0 else None
    prec  = tp / (tp + fp) if (tp + fp) > 0 else None
    f1    = (2 * prec * sens / (prec + sens)) if prec and sens and (prec + sens) > 0 else None
else:
    tn = fp = fn = tp = 0
    acc = sens = spec = prec = f1 = None

total = tn + fp + fn + tp


# ── KPI summary ────────────────────────────────────────────────────────────────
if mat is not None or df_bin_met is not None:
    section_title("Resumen binario")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Registros test", f"{total:,}", "total evaluados", accent="blue")
    with c2:
        metric_card("Accuracy",       f"{acc:.3f}"  if acc  is not None else "—",
                    "global",                              accent="teal")
    with c3:
        metric_card("Sensibilidad",   f"{sens:.3f}" if sens is not None else "—",
                    "recall Anormal",                      accent="warn")
    with c4:
        metric_card("Especificidad",  f"{spec:.3f}" if spec is not None else "—",
                    "recall Normal",                       accent="info")
    with c5:
        _f1_accent = "err" if (f1 is not None and f1 < 0.60) else "teal"
        metric_card("F1 Anormal",     f"{f1:.3f}"   if f1   is not None else "—",
                    "clase positiva",                      accent=_f1_accent)

    st.write("")


# ── Heatmap ────────────────────────────────────────────────────────────────────
section_title("Matriz de confusión 2×2")

if mat is not None:
    col_view, _ = st.columns([2.8, 7.2])
    with col_view:
        view = st.selectbox(
            "Vista",
            ["Conteos absolutos", "Normalizado por fila (%)"],
            index=0,
            key="cm_view_select",
        )

    disp_mat = mat.copy()
    fmt_str  = ",.0f"
    if view.startswith("Norm"):
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        disp_mat = mat / row_sums
        fmt_str  = ".1%"

    text_mat = [
        [f"TN\n{disp_mat[0,0]:{fmt_str}}", f"FP\n{disp_mat[0,1]:{fmt_str}}"],
        [f"FN\n{disp_mat[1,0]:{fmt_str}}", f"TP\n{disp_mat[1,1]:{fmt_str}}"],
    ]

    fig = go.Figure(go.Heatmap(
        z=disp_mat.tolist(),
        x=_LABELS,
        y=_LABELS,
        text=text_mat,
        texttemplate="%{text}",
        textfont=dict(size=13, family="IBM Plex Mono, monospace"),
        colorscale="Teal",
        colorbar=dict(title="", tickfont=dict(size=9, family="IBM Plex Mono, monospace")),
        hovertemplate=(
            "Real: <b>%{y}</b><br>"
            "Predicho: <b>%{x}</b><br>"
            "Valor: %{z}<extra></extra>"
        ),
    ))
    fig.update_layout(
        plot_bgcolor="#121a2b",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color="#8a98b5", size=11),
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Clase predicha", tickfont=dict(size=13), side="bottom"),
        yaxis=dict(title="Clase real",     tickfont=dict(size=13), autorange="reversed"),
    )

    col_hm, col_kv = st.columns([2.2, 1.8])
    with col_hm:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col_kv:
        with st.container(border=True):
            card_header("Valores absolutos", "test set completo")
            kv_table([
                (f'{badge("TN", "ok")} Real Normal · Pred Normal',     f"{tn:,}"),
                (f'{badge("FP", "warn")} Real Normal · Pred Anormal',  f"{fp:,}"),
                (f'{badge("FN", "err")} Real Anormal · Pred Normal',   f"{fn:,}"),
                (f'{badge("TP", "teal")} Real Anormal · Pred Anormal', f"{tp:,}"),
            ])
        if cm_png:
            with st.expander("Ver imagen PNG exportada", expanded=False):
                st.image(str(cm_png), use_container_width=True)

elif cm_png:
    callout(
        "info",
        "CSV no disponible — mostrando imagen exportada",
        "Para el heatmap interactivo ejecuta el pipeline y genera "
        "<code>reports/tables/confusion_matrix.csv</code>.",
    )
    st.image(str(cm_png), use_container_width=True)
else:
    callout(
        "warn",
        "Datos no disponibles",
        "No se encontró ningún archivo de matriz de confusión. "
        "Ejecuta el pipeline para generar <code>reports/tables/confusion_matrix.csv</code>.",
    )

st.write("")

# ── Métricas detalladas ────────────────────────────────────────────────────────
if acc is not None:
    section_title("Métricas derivadas")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        with st.container(border=True):
            card_header("Desempeño global", "test set · clase positiva = Anormal")
            kv_table([
                ("Accuracy",                        f"{acc:.3f}  ({acc:.1%})"),
                ("Sensibilidad / Recall Anormal",   f"{sens:.3f}  ({sens:.1%})" if sens is not None else "—"),
                ("Especificidad / Recall Normal",   f"{spec:.3f}  ({spec:.1%})" if spec is not None else "—"),
                ("Precision Anormal",               f"{prec:.3f}  ({prec:.1%})" if prec is not None else "—"),
                ("F1 Anormal",                      f"{f1:.3f}"                 if f1   is not None else "—"),
            ])

    with col_m2:
        with st.container(border=True):
            card_header("Errores del modelo", "conteos del test set")
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else None
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else None
            kv_table([
                ("FP — falsas alarmas",
                 f"{fp:,} &nbsp;({fp_rate:.1%} de los normales)" if fp_rate is not None else f"{fp:,}"),
                ("FN — casos no detectados",
                 f"{fn:,} &nbsp;({fn_rate:.1%} de los anormales)" if fn_rate is not None else f"{fn:,}"),
                ("TN — normales correctos",  f"{tn:,}"),
                ("TP — anormales detectados", f"{tp:,}"),
            ])

    st.write("")

# ── Interpretación ─────────────────────────────────────────────────────────────
section_title("Interpretación")

with st.container(border=True):
    card_header("Lectura del resultado binario", "contexto académico")

    _tn_tp  = tn + tp
    _sens_s = f"{sens:.1%}" if sens is not None else "—"
    _spec_s = f"{spec:.1%}" if spec is not None else "—"
    _acc_s  = f"{acc:.1%}"  if acc  is not None else "—"

    st.html(
        f'<div style="font-size:13px;color:var(--fg-1);line-height:1.8;max-width:860px">'
        f"<p>La <b>diagonal principal</b> (TN + TP = {_tn_tp:,}) representa los aciertos: "
        f"{tn:,} registros normales y {tp:,} anormales clasificados correctamente. "
        f"La <b>accuracy global</b> es {_acc_s}.</p>"
        f"<p>Los <b>FN ({fn:,})</b> son registros anormales clasificados como normales — "
        f"el error más crítico en contexto clínico porque corresponde a arritmias no detectadas. "
        f"La <b>sensibilidad de {_sens_s}</b> significa que el modelo detecta aproximadamente "
        f"ese porcentaje de las anormalidades reales.</p>"
        f"<p>Los <b>FP ({fp:,})</b> son registros normales marcados como anormales (falsas alarmas). "
        f"La <b>especificidad de {_spec_s}</b> indica que el modelo clasifica correctamente "
        f"ese porcentaje de los registros normales.</p>"
        f"<p>El modelo tiene <b>mejor especificidad que sensibilidad</b>: reconoce mejor "
        f"lo normal que lo anormal. Para el objetivo de detección de arritmias, "
        f"mejorar la sensibilidad sería el paso prioritario.</p>"
        f"</div>"
    )

callout(
    "warn",
    "Demo académica — no para uso clínico",
    "Resultados del test set del pipeline académico sobre VitalDB. "
    "<b>No deben interpretarse como rendimiento clínico real.</b>",
)

page_footer()
