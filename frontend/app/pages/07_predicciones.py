"""Página 07 — Demo binaria normal/anormal.

Evalúa casos VitalDB con el modelo tabular binario (LinearSVC).
Modo A: .npy disponible → ECG + evaluación completa.
Modo B: sin .npy → evaluación tabular pre-computada o desde parquet local.

Esta demo NO constituye diagnóstico clínico.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.paths import (
    PROJECT_ROOT, DATA_DIR,
    DEMO_CASES_CSV, ARTIFACTS_NPY_DIR,
    resolve_npy_dir,
)
from components.badges import badge
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.layout import page_header, page_footer
from utils.loaders import (
    load_model,
    load_model_metadata,
    load_binary_case_level_metrics,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PARQUET_PATH = DATA_DIR / "processed" / "filtered_tabular_modeling_dataset.parquet"

_LABEL_DISPLAY = {"normal": "Normal", "abnormal": "Anormal"}
_BINARY_LABELS  = ("normal", "abnormal")

_FEAT_COLS_FALLBACK: list[str] = [
    "time_second", "analyzed_duration_sec", "total_beats", "caseend", "anestart",
    "aneend", "opstart", "opend", "height", "weight", "bmi", "preop_plt", "preop_pt",
    "preop_k", "preop_alb", "preop_ast", "preop_alt", "preop_cr", "tubesize",
    "intraop_uo", "intraop_crystalloid", "intraop_rocu", "rr_prev", "rr_next",
    "hr_inst_from_rr_prev", "position_in_case", "optype", "iv1", "aline1", "cline1",
]

# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------
def _to_binary(labels) -> np.ndarray:
    return np.array([
        "normal" if str(lbl).strip() == "N" else "abnormal"
        for lbl in np.asarray(labels).astype(str)
    ])


def _normalize_pred(pred) -> str:
    s = str(pred).strip().lower()
    return "normal" if s in ("normal", "0", "n", "false") else "abnormal"


def _normalize_preds(preds: np.ndarray) -> np.ndarray:
    return np.array([_normalize_pred(p) for p in preds])


def _display(val: str) -> str:
    return _LABEL_DISPLAY.get(str(val).lower().strip(), str(val).title())


# ---------------------------------------------------------------------------
# Slider helpers — prevent StreamlitAPIException when value is out of range
# ---------------------------------------------------------------------------
def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _clamp(v, lo, hi) -> float:
    lo = _safe_float(lo, 0.0)
    hi = _safe_float(hi, lo)
    if hi < lo:
        hi = lo
    return max(lo, min(_safe_float(v, lo), hi))


def _safe_slider(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float = 1.0,
    key: str | None = None,
) -> float:
    """Render st.slider with guaranteed valid range; returns min_value if range is degenerate."""
    min_value = _safe_float(min_value, 0.0)
    max_value = _safe_float(max_value, min_value)
    if max_value <= min_value:
        return min_value
    value = _clamp(value, min_value, max_value)
    # Clear stale session-state values that fall outside the current range
    if key is not None and key in st.session_state:
        try:
            old = float(st.session_state[key])
            if old < min_value or old > max_value:
                del st.session_state[key]
        except Exception:
            del st.session_state[key]
    return float(st.slider(
        label,
        min_value=float(min_value),
        max_value=float(max_value),
        value=float(value),
        step=float(step),
        key=key,
    ))


def _evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, balanced_accuracy_score,
    )
    mask = ~np.isin(y_true, ["nan", "None", "none", ""])
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {}
    has_both = len(set(yt)) == 2
    kw = dict(pos_label="abnormal", zero_division=0)
    return {
        "n":           int(len(yt)),
        "accuracy":    float(accuracy_score(yt, yp)),
        "balanced":    float(balanced_accuracy_score(yt, yp)) if has_both else None,
        "precision":   float(precision_score(yt, yp, **kw)),
        "recall":      float(recall_score(yt, yp, **kw)),
        "f1":          float(f1_score(yt, yp, **kw)),
        "specificity": float(recall_score(yt, yp, pos_label="normal", zero_division=0)),
        "cm":          confusion_matrix(yt, yp, labels=list(_BINARY_LABELS)) if has_both else None,
    }


# ---------------------------------------------------------------------------
# Demo case catalogue
# ---------------------------------------------------------------------------
_DEMO_CASES_FALLBACK: list[dict] = [
    {
        "case_id": 5377, "title": "Normal estable", "binary_type": "normal",
        "description": "Caso casi completamente normal. 1165/1169 registros normales. "
                       "El modelo lo clasifica correctamente.",
        "expected_pattern": "Mayoría normales", "n_beats": 1169,
        "accuracy": 1.0, "recall_abnormal": 1.0, "notes": "",
    },
    {
        "case_id": 2040, "title": "Anormal claro", "binary_type": "abnormal",
        "description": "Caso 100% anormal detectado correctamente por el modelo.",
        "expected_pattern": "Todos anormales", "n_beats": 1045,
        "accuracy": 1.0, "recall_abnormal": 1.0, "notes": "",
    },
    {
        "case_id": 337, "title": "Mixto representativo", "binary_type": "mixed",
        "description": "Proporción balanceada entre normal y anormal. "
                       "El modelo alcanza accuracy de 0.784.",
        "expected_pattern": "Mezcla normal/anormal", "n_beats": 925,
        "accuracy": 0.784, "recall_abnormal": 0.811, "notes": "",
    },
    {
        "case_id": 1996, "title": "Mixto adicional", "binary_type": "mixed",
        "description": "Caso con presencia de registros Normal y Anormal. "
                       "El modelo alcanza accuracy de 0.754 y recall anormal de 0.771.",
        "expected_pattern": "Normal + Anormal", "n_beats": 724,
        "accuracy": 0.754, "recall_abnormal": 0.771, "notes": "",
    },
]

_BINARY_TYPE_META = {
    "normal":   {"icon": "✓",  "accent": "teal", "label": "Normal"},
    "abnormal": {"icon": "⚠",  "accent": "err",  "label": "Anormal"},
    "mixed":    {"icon": "📊", "accent": "blue",  "label": "Mixto"},
}


def _load_demo_cases() -> list[dict]:
    if DEMO_CASES_CSV.exists():
        try:
            df = pd.read_csv(DEMO_CASES_CSV)
            if "n_beats" in df.columns:
                df["n_beats"] = pd.to_numeric(df["n_beats"], errors="coerce")
            return df.to_dict(orient="records")
        except Exception:
            pass
    return _DEMO_CASES_FALLBACK


_DEMO_CASES    = _load_demo_cases()
_DEMO_FRAG_DIR = resolve_npy_dir()


# ---------------------------------------------------------------------------
# Mode B — tabular evaluation without ECG signal
# ---------------------------------------------------------------------------
def _render_mode_b(case_id: int) -> None:
    dc    = next((d for d in _DEMO_CASES if int(d["case_id"]) == case_id), {})
    btype = str(dc.get("binary_type", "mixed")).lower()
    tmeta = _BINARY_TYPE_META.get(btype, _BINARY_TYPE_META["mixed"])

    # Callout: no ECG
    callout(
        "info",
        "Señal ECG no disponible para este caso",
        "El archivo <code>.npy</code> no está incluido en los artefactos de despliegue. "
        "Se muestra la <b>evaluación tabular</b> del modelo usando las features procesadas "
        "(RR intervals + metadatos clínicos). "
        "Los resultados son idénticos a los obtenidos en la evaluación oficial.",
    )

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Get full per-case metrics from binary_case_level_metrics.csv
    df_all  = load_binary_case_level_metrics()
    cm_row: pd.Series | None = None
    if df_all is not None and "case_id" in df_all.columns:
        sub    = df_all[df_all["case_id"] == case_id]
        cm_row = sub.iloc[0] if not sub.empty else None

    def _m(col: str, default=None):
        if cm_row is not None and col in cm_row.index:
            v = cm_row[col]
            return None if (isinstance(v, float) and np.isnan(v)) else v
        return dc.get(col, default)

    n_total    = int(_m("n_records",   dc.get("n_beats") or 0))
    n_normal   = int(_m("n_normal",    0))
    n_abnormal = int(_m("n_abnormal",  0))
    pct_n      = float(_m("pct_normal",   n_normal  / max(n_total, 1)))
    pct_ab     = float(_m("pct_abnormal", n_abnormal / max(n_total, 1)))

    acc     = _m("accuracy")
    bal     = _m("balanced_accuracy")
    prec_ab = _m("precision_abnormal")
    rec_ab  = _m("recall_abnormal")
    f1_ab   = _m("f1_abnormal")
    spec_n  = _m("specificity_normal")

    # Case header card
    with st.container(border=True):
        card_header(
            dc.get("title", f"case_{case_id}"),
            f"case_id = {case_id} · {tmeta['label']}",
            right_html=badge(tmeta["label"], tmeta["accent"]),
        )
        st.markdown(
            f'<p style="font-size:13px;color:var(--fg-1);margin:6px 0 0">'
            f'{dc.get("description","")}</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    # Distribution
    section_title("Distribución real del caso")

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        metric_card("Total registros", f"{n_total:,}", accent="blue")
    with col_d2:
        metric_card("Normal",  f"{n_normal:,}",   helper=f"{pct_n:.1%}",  accent="teal")
    with col_d3:
        metric_card("Anormal", f"{n_abnormal:,}", helper=f"{pct_ab:.1%}", accent="err")
    with col_d4:
        metric_card("Clase dominante",
                    "Normal" if n_normal >= n_abnormal else "Anormal",
                    accent="blue")

    if n_total > 0:
        import plotly.graph_objects as go
        fig_bar = go.Figure()
        fig_bar.add_bar(
            x=["Normal", "Anormal"],
            y=[n_normal, n_abnormal],
            marker_color=["#2dd4bf", "#f87171"],
            text=[f"{n_normal:,} ({pct_n:.1%})", f"{n_abnormal:,} ({pct_ab:.1%})"],
            textposition="outside",
        )
        fig_bar.update_layout(
            height=220, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c5cfe0", showlegend=False,
            yaxis=dict(showgrid=False, visible=False),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Metrics
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    section_title("Métricas del modelo en este caso")

    def _fmt(v) -> str:
        return "—" if v is None else f"{float(v):.1%}"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        metric_card("Accuracy",           _fmt(acc),     accent="teal")
    with m2:
        metric_card("Balanced acc.",      _fmt(bal),     accent="teal")
    with m3:
        metric_card("Recall Anormal",     _fmt(rec_ab),  accent="err")
    with m4:
        metric_card("Precision Anormal",  _fmt(prec_ab), accent="err")
    with m5:
        metric_card("F1 Anormal",         _fmt(f1_ab),   accent="warn")
    with m6:
        metric_card("Specificity Normal", _fmt(spec_n),  accent="blue")

    with st.container(border=True):
        card_header("Detalle de la evaluación", f"case_id = {case_id}")
        kv_rows = [
            ("Caso",              f"case_id = {case_id}"),
            ("Tipo",              tmeta["label"]),
            ("Total registros",   f"{n_total:,}"),
            ("Normal",            f"{n_normal:,} ({pct_n:.1%})"),
            ("Anormal",           f"{n_abnormal:,} ({pct_ab:.1%})"),
        ]
        if acc  is not None: kv_rows.append(("Accuracy",
                                              f'<b style="color:var(--teal)">{float(acc):.3f}</b>'))
        if bal  is not None: kv_rows.append(("Balanced accuracy",  f"{float(bal):.3f}"))
        if rec_ab  is not None: kv_rows.append(("Recall Anormal",   f"{float(rec_ab):.3f}"))
        if prec_ab is not None: kv_rows.append(("Precision Anormal",f"{float(prec_ab):.3f}"))
        if f1_ab   is not None: kv_rows.append(("F1 Anormal",       f"{float(f1_ab):.3f}"))
        if spec_n  is not None: kv_rows.append(("Specificity Normal",f"{float(spec_n):.3f}"))
        kv_table(kv_rows)

    # Live prediction from parquet (local only)
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    section_title("Predicciones individuales")

    if not _PARQUET_PATH.exists():
        callout(
            "info",
            "Dataset tabular no disponible en este entorno",
            "El parquet no está incluido en los artefactos de despliegue (~600 MB). "
            "Las métricas del resumen provienen de la evaluación ejecutada localmente.",
        )
        return

    _model_p = load_model()
    _meta_md = load_model_metadata()
    if _model_p is None:
        callout("warn", "Modelo no disponible",
                "No se encontró <code>tabular_best_model_pipeline.joblib</code>.")
        return

    feat_cols = (
        (_meta_md.get("numeric_features", []) + _meta_md.get("categorical_features", []))
        if _meta_md else _FEAT_COLS_FALLBACK
    )

    with st.spinner(f"Cargando registros de case_id={case_id}…"):
        try:
            df_case = pd.read_parquet(
                _PARQUET_PATH,
                filters=[("case_id", "=", case_id)],
            )
        except Exception as e:
            callout("err", "Error al leer el parquet", str(e))
            return

    if df_case.empty:
        callout("warn", "Caso no encontrado",
                f"case_id={case_id} no tiene filas en el dataset tabular.")
        return

    y_true_live = _to_binary(df_case["rhythm_label"].values) \
        if "rhythm_label" in df_case.columns else None
    try:
        X_live   = df_case[[c for c in feat_cols if c in df_case.columns]]
        y_pred_live = _normalize_preds(_model_p.predict(X_live))
    except Exception as e:
        callout("err", "Error en predicción", str(e))
        return

    show_cols = [c for c in ["time_second", "rhythm_label", "beat_type"] if c in df_case.columns]
    res_df = df_case[show_cols].copy().reset_index(drop=True)
    res_df["Real"]      = [_display(v) for v in (y_true_live if y_true_live is not None else ["—"] * len(y_pred_live))]
    res_df["Predicción"] = [_display(v) for v in y_pred_live]
    if y_true_live is not None:
        _correct_mask = pd.Series(
            np.asarray(y_true_live).astype(str) == np.asarray(y_pred_live).astype(str),
            index=res_df.index,
        )
        res_df["✓/✗"] = _correct_mask.map({True: "✓", False: "✗"})
    if "time_second" in res_df.columns:
        res_df["time_second"] = res_df["time_second"].round(3)

    col_filt, _ = st.columns([2, 6])
    with col_filt:
        _show_errors = st.checkbox("Mostrar solo errores", value=False, key="b_mode_errors")

    res_display = (
        res_df[res_df["✓/✗"] == "✗"] if _show_errors and "✓/✗" in res_df.columns
        else res_df
    )

    with st.container(border=True):
        card_header(
            "Real vs Predicción",
            f"{len(res_display):,} / {len(res_df):,} filas · Normal / Anormal",
        )
        col_cfg: dict = {
            "time_second": st.column_config.NumberColumn("time (s)", format="%.3f", width="small"),
            "rhythm_label": st.column_config.TextColumn("rhythm_label (original)", width="medium"),
        }
        st.dataframe(
            res_display.head(400),
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
        )
        st.caption(
            "rhythm_label es contexto — la evaluación binaria es "
            "Normal (N) vs Anormal (≠ N). "
            f"Mostrando {min(len(res_display), 400):,} de {len(res_display):,} filas."
        )

    if y_true_live is not None and "✓/✗" in res_df.columns:
        n_err = (res_df["✓/✗"] == "✗").sum()
        if n_err > 0:
            with st.expander(f"Errores de clasificación ({n_err:,})", expanded=False):
                err_grp = (
                    res_df[res_df["✓/✗"] == "✗"]
                    .groupby(["Real", "Predicción"])
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                st.dataframe(err_grp, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Page header & callout
# ---------------------------------------------------------------------------
page_header(
    "Clasificación binaria: Normal / Anormal",
    "Demo con el modelo LinearSVC — detección de ritmos intraoperatorios.",
    badge_html=badge("Demo binaria", "info"),
)

callout(
    "info",
    "Modelo binario — Normal vs Anormal",
    "El modelo clasifica cada latido como <b>Normal</b> (rhythm_label = N) o "
    "<b>Anormal</b> (cualquier otra etiqueta). "
    "Usa <b>features tabulares</b>: RR intervals + metadata clínica. "
    "La señal ECG se muestra cuando está disponible — no alimenta el modelo directamente. "
    "<br><span style='color:var(--fg-3)'>Demo académica — no constituye diagnóstico clínico.</span>",
)

st.write("")

# ---------------------------------------------------------------------------
# Section 1: Demo case grid
# ---------------------------------------------------------------------------
section_title("Casos VitalDB demo")

st.html(
    '<p style="font-size:12px;color:var(--fg-3);margin:0 0 12px">'
    'Casos seleccionados para ilustrar distintos patrones binarios. '
    'Haz clic en un caso para cargarlo.'
    '</p>'
)

_cols = st.columns(len(_DEMO_CASES))
for _i, _dc in enumerate(_DEMO_CASES):
    with _cols[_i]:
        _cid   = int(_dc["case_id"])
        _btype = str(_dc.get("binary_type", "mixed")).lower()
        _tmeta = _BINARY_TYPE_META.get(_btype, _BINARY_TYPE_META["mixed"])
        _is_active = st.session_state.get("_p7_active_case") == _cid

        _npy_avail  = (_DEMO_FRAG_DIR / f"case_{_cid}.npy").exists()
        _npy_badge  = (
            '<span style="font-size:9px;color:var(--teal)">● señal ECG disponible</span>'
            if _npy_avail
            else '<span style="font-size:9px;color:var(--fg-3)">○ evaluación tabular</span>'
        )
        _n_beats    = _dc.get("n_beats")
        try:
            _beats_str = f"{int(float(_n_beats)):,} registros" if _n_beats else "registros: —"
        except (TypeError, ValueError):
            _beats_str = "registros: —"

        _border = "2px solid var(--teal)" if _is_active else "1px solid var(--line-1)"
        _title  = str(_dc.get("title", f"case_{_cid}"))
        _desc   = str(_dc.get("description", ""))

        st.html(
            f'<div style="border:{_border};border-radius:8px;padding:10px 10px 8px;'
            f'background:var(--bg-1);min-height:148px">'
            f'<div style="font-size:18px;margin-bottom:4px">{_tmeta["icon"]}</div>'
            f'<div style="font-size:10px;font-weight:700;color:var(--{_tmeta["accent"]});'
            f'letter-spacing:.04em;margin-bottom:3px;text-transform:uppercase">{_tmeta["label"]}</div>'
            f'<div style="font-size:11px;font-weight:600;color:var(--fg-0);'
            f'line-height:1.3;margin-bottom:5px">{_title}</div>'
            f'<div style="font-size:10px;color:var(--fg-3);margin-bottom:6px;line-height:1.4">'
            f'{_desc[:70]}{"…" if len(_desc) > 70 else ""}</div>'
            f'<div style="font-size:10px;font-family:var(--mono);color:var(--fg-3)">'
            f'case_id {_cid} · {_beats_str}</div>'
            f'<div style="margin-top:5px">{_npy_badge}</div>'
            f'</div>'
        )
        if st.button("Cargar", key=f"load_case_{_cid}", use_container_width=True):
            st.session_state["_p7_active_case"] = _cid
            st.session_state["_p7_signal"]       = None
            st.session_state["_p7_proc_signal"]  = None
            st.session_state["_p7_feat_df"]      = None
            st.session_state["_p7_predictions"]  = None
            st.session_state["_p7_source"]       = "demo"
            st.rerun()


st.write("")

# ---------------------------------------------------------------------------
# Section 3: Upload .npy
# ---------------------------------------------------------------------------
section_title("Explorar señal ECG (.npy)")

callout(
    "info",
    "Demostración conceptual — carga de señal ECG",
    "Esta sección muestra cómo podría funcionar una carga directa de señales ECG "
    "en una versión futura de la app. "
    "El modelo actual <b>no predice únicamente a partir del archivo .npy</b>, "
    "porque fue entrenado con features tabulares derivadas de anotaciones, "
    "intervalos RR y metadata clínica. "
    "Por eso, los archivos ECG externos se usan para <b>visualización</b>. "
    "Solo los casos VitalDB conocidos que tengan features procesadas disponibles "
    "pueden evaluarse con el modelo binario Normal/Anormal.",
)

st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

st.html(
    '<p style="font-size:12px;color:var(--fg-3);margin:0 0 10px">'
    'Puedes subir un archivo <code>.npy</code> 1D para visualizar la señal ECG. '
    'Si el nombre contiene un case_id disponible (p. ej. <code>case_337.npy</code> '
    'o <code>case_337 (1).npy</code>), la app intentará asociarlo con sus features '
    'tabulares y mostrar predicciones. '
    'Para señales externas sin features asociadas, solo se mostrará la visualización.'
    '</p>'
)
_uploaded = st.file_uploader(
    "Seleccionar archivo .npy",
    type=["npy"],
    key="npy_uploader",
    label_visibility="collapsed",
)

if _uploaded is not None:
    _upload_cid_detected = None
    try:
        from utils.case_eval import extract_case_id_from_filename
        _upload_cid_detected = extract_case_id_from_filename(_uploaded.name)
    except Exception:
        pass
    if st.session_state.get("_p7_last_upload") != _uploaded.name:
        st.session_state["_p7_last_upload"] = _uploaded.name
        st.session_state["_p7_active_case"] = _upload_cid_detected
        st.session_state["_p7_signal"]      = None
        st.session_state["_p7_proc_signal"] = None
        st.session_state["_p7_feat_df"]     = None
        st.session_state["_p7_predictions"] = None
        st.session_state["_p7_source"]      = "upload"
        st.session_state["_p7_upload_file"] = _uploaded
        st.rerun()

# ---------------------------------------------------------------------------
# Resolve active case and npy availability
# ---------------------------------------------------------------------------
_active_cid = st.session_state.get("_p7_active_case")
_source     = st.session_state.get("_p7_source", "demo")
_upload_obj = st.session_state.get("_p7_upload_file") if _source == "upload" else None

_frag_p    = _DEMO_FRAG_DIR / f"case_{_active_cid}.npy" if _active_cid else None
_npy_exists = bool(
    (_frag_p is not None and _frag_p.exists())
    or (_source == "upload" and _upload_obj is not None)
)

# Try importing ECG signal modules (only needed for Mode A — don't stop on failure)
try:
    from utils.case_eval import (
        find_annotation_file, load_case_annotations, load_case_metadata,
        get_case_features_from_parquet, get_case_features_for_demo,
        demo_case_features_exist, build_tabular_features_from_case,
        load_npy_signal, summarize_npy_signal, find_valid_segments,
        predict_case_windows, TARGET_FS, WAVEFORMS_DIR, PARQUET_PATH,
    )
    from components.ecg_viewer import (
        plot_ecg_signal, plot_raw_vs_processed,
        plot_annotations_on_ecg, plot_ecg_with_prediction_regions,
        plot_ecg_with_binary_prediction_bands,
    )
    _EVAL_MODULES_OK = True
    # Also check full waveform path if modules loaded
    if _active_cid and not _npy_exists:
        _npy_exists = (WAVEFORMS_DIR / f"case_{_active_cid}.npy").exists()
except ImportError:
    _EVAL_MODULES_OK = False

# ---------------------------------------------------------------------------
# No case selected → placeholder
# ---------------------------------------------------------------------------
if _active_cid is None:
    st.html(
        '<div class="placeholder-block" style="min-height:120px;padding:24px">'
        '<div class="ph-mono">sin caso activo</div>'
        '<div class="ph-title">Selecciona un caso demo o sube un archivo .npy</div>'
        '</div>'
    )
    st.stop()

# ---------------------------------------------------------------------------
# Mode B — no .npy: tabular evaluation
# ---------------------------------------------------------------------------
if not _npy_exists:
    st.html('<hr style="border-color:var(--line-1);margin:4px 0 20px">')
    st.html(
        f'<div style="display:inline-flex;align-items:center;gap:10px;padding:6px 14px;'
        f'background:rgba(74,140,255,0.08);border:1px solid rgba(74,140,255,0.3);'
        f'border-radius:6px;margin-bottom:14px">'
        f'<span style="color:#4a8cff;font-weight:600;font-size:13px">📊 Evaluación tabular</span>'
        f'<span style="color:#8a98b5;font-size:11px;font-family:var(--mono)">'
        f'case_id = {_active_cid}</span>'
        f'<span style="color:#8a98b5;font-size:11px">· señal ECG no disponible</span>'
        f'</div>'
    )
    _render_mode_b(_active_cid)
    callout(
        "warn",
        "Demo académica — no para uso clínico",
        "Resultados sobre un caso individual. "
        "Métricas oficiales del modelo: test_f1_macro=0.615, accuracy=0.633 "
        "(conjunto de test completo).",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Mode A — .npy available
# ---------------------------------------------------------------------------
if not _EVAL_MODULES_OK:
    callout(
        "err",
        "Módulos de señal no disponibles",
        "No se pueden cargar los módulos de procesamiento ECG. "
        "Verifica las dependencias en <code>requirements.txt</code>.",
    )
    st.stop()

if st.session_state.get("_p7_signal") is None:
    _sig_path = None
    if _source == "demo":
        _full_path = WAVEFORMS_DIR / f"case_{_active_cid}.npy"
        _sig_path  = _full_path if _full_path.exists() else (_frag_p if _frag_p and _frag_p.exists() else None)
    elif _source == "upload" and _upload_obj is not None:
        _upload_obj.seek(0)
        _sig_path = _upload_obj

    if _sig_path is not None:
        try:
            st.session_state["_p7_signal"] = load_npy_signal(_sig_path)
        except Exception as _load_err:
            callout("err", "Error al cargar el archivo .npy", str(_load_err))

_signal = st.session_state.get("_p7_signal")
if _signal is None:
    st.warning("No se pudo cargar la señal. Verifica el archivo o selecciona otro caso.")
    st.stop()

# Determine feature availability (works in Streamlit Cloud via demo parquets)
_ann_path    = find_annotation_file(_active_cid) if _active_cid else None
_meta_md_a   = load_model_metadata()
_feat_cols_a = (
    (_meta_md_a.get("numeric_features", []) + _meta_md_a.get("categorical_features", []))
    if _meta_md_a else _FEAT_COLS_FALLBACK
)
# Case A if any features source is reachable for this case_id
_has_features_src = (
    _active_cid is not None
    and _feat_cols_a
    and (
        demo_case_features_exist(_active_cid)   # demo parquet in app_artifacts ✓ cloud
        or PARQUET_PATH.exists()                 # full parquet — local dev only
        or _ann_path is not None                 # build from annotation + metadata
    )
)
_is_case_a = _has_features_src

st.write("")
st.html('<hr style="border-color:var(--line-1);margin:4px 0 20px">')

if _is_case_a:
    _feat_src = (
        "demo parquet" if demo_case_features_exist(_active_cid)
        else "parquet local" if PARQUET_PATH.exists()
        else "anotaciones"
    )
    _src_note = (
        f"· features tabulares disponibles ({_feat_src}) · predicción binaria disponible"
        if _source == "upload"
        else f"· features tabulares ({_feat_src}) · predicción binaria disponible"
    )
    st.html(
        f'<div style="display:inline-flex;align-items:center;gap:10px;padding:6px 14px;'
        f'background:rgba(45,212,191,0.08);border:1px solid rgba(45,212,191,0.3);'
        f'border-radius:6px;margin-bottom:14px">'
        f'<span style="color:#2dd4bf;font-weight:600;font-size:13px">✓ Caso VitalDB reconocido</span>'
        f'<span style="color:#8a98b5;font-size:11px;font-family:var(--mono)">case_id = {_active_cid}</span>'
        f'<span style="color:#8a98b5;font-size:11px">{_src_note}</span>'
        f'</div>'
    )
    if _source == "upload":
        callout(
            "info",
            f"Se detectó case_id = {_active_cid} y se encontraron features tabulares asociadas",
            "La señal <code>.npy</code> se usa para visualización, mientras que la predicción "
            "Normal/Anormal se calcula con las features procesadas del caso. "
            "Esta funcionalidad es demostrativa y académica. "
            "No debe usarse para diagnóstico ni interpretación clínica real.",
        )
else:
    _no_feat_note = (
        f"case_id {_active_cid} detectado — features tabulares no disponibles"
        if _active_cid is not None
        else "Señal ECG externa — solo visualización"
        if _source == "upload"
        else "Archivo sin features asociadas — solo visualización ECG"
    )
    st.html(
        f'<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;'
        f'background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);'
        f'border-radius:6px;margin-bottom:14px">'
        f'<span style="color:#fbbf24;font-weight:600;font-size:13px">⚠ {_no_feat_note}</span>'
        f'<span style="color:#8a98b5;font-size:11px">predicción no disponible para este archivo</span>'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Section 4: Signal summary
# ---------------------------------------------------------------------------
section_title("Resumen de la señal")

_summary = summarize_npy_signal(_signal, fs=TARGET_FS)
_segs    = find_valid_segments(_signal, fs=TARGET_FS, min_duration_s=5.0, max_segments=5)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Muestras",          f"{_summary['n_samples']:,}")
c2.metric("Duración (s)",      f"{_summary['duration_s']:,.0f}")
c3.metric("fs asumida",        f"{int(TARGET_FS)} Hz")
c4.metric("NaN",               f"{_summary['nan_pct']:.1f}%")
c5.metric("Segmentos válidos", str(len(_segs)))

_segs and (_seg_start := _segs[0][0]) and (_seg_end := _segs[0][1])
if _segs:
    _seg_start, _seg_end = _segs[0]
    _disp_offset = _seg_start / TARGET_FS
    _disp_sig    = _signal[_seg_start:_seg_end + 1]
else:
    _disp_offset = 0.0
    _disp_sig    = _signal

# ---------------------------------------------------------------------------
# Section 5: ECG crudo
# ---------------------------------------------------------------------------
section_title("Señal ECG cruda")
st.caption("Visualización del archivo .npy. Esta señal **no alimenta el modelo**.")
_fname_label = f"case_{_active_cid}.npy" if _active_cid else "archivo .npy"
st.plotly_chart(
    plot_ecg_signal(
        _disp_sig, TARGET_FS,
        title=f"ECG crudo · {_fname_label} · primeros 10 s del segmento válido",
        start_time_offset=_disp_offset, max_seconds=10.0,
    ),
    use_container_width=True, config={"displayModeBar": False},
)

# ---------------------------------------------------------------------------
# Section 6: ECG preprocesado
# ---------------------------------------------------------------------------
section_title("Preprocesamiento ECG")

if st.button("Preprocesar señal (filtro 0.5–40 Hz · normalización)", key="btn_preproc"):
    with st.spinner("Procesando…"):
        try:
            from src.pipeline import preprocess_ecg
            st.session_state["_p7_proc_signal"] = preprocess_ecg(_disp_sig, original_fs=TARGET_FS)
            st.session_state["_p7_proc_error"]  = None
        except Exception as _pe:
            st.session_state["_p7_proc_error"]  = str(_pe)
            st.session_state["_p7_proc_signal"] = None

_proc_sig = st.session_state.get("_p7_proc_signal")
_proc_err = st.session_state.get("_p7_proc_error")
if _proc_err:
    st.error(_proc_err)
elif _proc_sig is not None:
    _fr, _fp = plot_raw_vs_processed(
        _disp_sig, _proc_sig, raw_fs=TARGET_FS,
        start_time_offset=_disp_offset, max_seconds=10.0,
    )
    col_r, col_p = st.columns(2)
    with col_r:
        st.plotly_chart(_fr, use_container_width=True, config={"displayModeBar": False})
    with col_p:
        st.plotly_chart(_fp, use_container_width=True, config={"displayModeBar": False})

# ===========================================================================
# CASO A — prediction + binary evaluation (features available)
# ===========================================================================
if _is_case_a:
    st.html('<hr style="border-color:var(--line-1);margin:24px 0 16px">')
    section_title(f"Predicción binaria · case_id = {_active_cid}")

    # Note: _meta_md_a and _feat_cols_a already computed above the banner block
    _model_p  = load_model()

    # ── Load feature DataFrame (cloud-friendly: demo parquet > full parquet > build)
    _feat_df = st.session_state.get("_p7_feat_df")
    if _feat_df is None:
        _feat_df = get_case_features_for_demo(_active_cid, _feat_cols_a)

        # Fallback: build from annotation file + metadata (local only)
        if _feat_df is None and _ann_path is not None:
            _ann_raw  = load_case_annotations(_ann_path)
            _meta_row = load_case_metadata(_active_cid)
            if _meta_row is not None and len(_ann_raw) > 0:
                _feat_df, _missing = build_tabular_features_from_case(
                    _ann_raw, _meta_row, _feat_cols_a
                )
                if _missing:
                    callout("warn", f"{len(_missing)} features faltantes",
                            ", ".join(f"<code>{f}</code>" for f in _missing))

        st.session_state["_p7_feat_df"] = _feat_df

    # ── Distribution metrics from feat_df (works without annotation file)
    if _feat_df is not None and len(_feat_df) > 0:
        _has_rl     = "rhythm_label" in _feat_df.columns
        _n_records  = len(_feat_df)
        _n_normal   = int((_feat_df["rhythm_label"] == "N").sum())  if _has_rl else None
        _n_abnormal = (_n_records - _n_normal)                      if _n_normal is not None else None
        _n_classes  = int(_feat_df["rhythm_label"].nunique())        if _has_rl else None

        col_ann1, col_ann2, col_ann3, col_ann4 = st.columns(4)
        col_ann1.metric("Registros", f"{_n_records:,}")
        col_ann2.metric("Normal (N)",    f"{_n_normal:,}"   if _n_normal   is not None else "—")
        col_ann3.metric("Anormal (≠ N)", f"{_n_abnormal:,}" if _n_abnormal is not None else "—")
        col_ann4.metric("Clases originales", str(_n_classes) if _n_classes else "—")

    # ── ECG annotation overlay (only if annotation CSV is available locally)
    if _ann_path is not None and _feat_df is not None:
        _ann_overlay = load_case_annotations(_ann_path)
        _ann_win = _ann_overlay[
            (_ann_overlay["time_second"] >= _disp_offset) &
            (_ann_overlay["time_second"] <= _disp_offset + 10)
        ]
        if len(_ann_win) > 0:
            with st.container(border=True):
                card_header(
                    "ECG con latidos anotados",
                    f"{len(_ann_win)} latidos en ventana de 10 s · "
                    "colores por rhythm_label (contexto, no salida del modelo)",
                )
                st.plotly_chart(
                    plot_annotations_on_ecg(
                        _disp_sig, _ann_win, fs=TARGET_FS,
                        start_time_offset=_disp_offset, max_seconds=10.0,
                    ),
                    use_container_width=True, config={"displayModeBar": False},
                )

    if _model_p is None:
        callout("warn", "Modelo no disponible",
                "No se encontró <code>tabular_best_model_pipeline.joblib</code>.")
    elif _feat_df is None or len(_feat_df) == 0:
        callout("warn", "Features no disponibles",
                f"No se pudieron obtener features tabulares para case_id={_active_cid}. "
                "Sin RR intervals y metadata clínica no es posible ejecutar el modelo.")
    else:
        _preds_raw = st.session_state.get("_p7_predictions")
        if _preds_raw is None:
            try:
                _preds_raw = predict_case_windows(_feat_df, _model_p, _feat_cols_a)
                st.session_state["_p7_predictions"] = _preds_raw
            except Exception as _pe:
                callout("err", "Error en predicción", str(_pe))
                _preds_raw = None

        if _preds_raw is not None:
            _y_pred_bin = _normalize_preds(_preds_raw)
            _y_true_str = (
                _feat_df["rhythm_label"].to_numpy()
                if "rhythm_label" in _feat_df.columns else None
            )
            _y_true_bin = _to_binary(_y_true_str) if _y_true_str is not None else None
            _eval = (
                _evaluate_binary(_y_true_bin, _y_pred_bin)
                if _y_true_bin is not None else None
            )

            st.write("")
            section_title("Resumen del caso evaluado")

            _n_total     = len(_y_pred_bin)
            _n_pred_norm = int(np.sum(_y_pred_bin == "normal"))
            _n_pred_abn  = int(np.sum(_y_pred_bin == "abnormal"))
            _n_real_norm = int(np.sum(_y_true_bin == "normal"))  if _y_true_bin is not None else None
            _n_real_abn  = int(np.sum(_y_true_bin == "abnormal")) if _y_true_bin is not None else None

            # ── Composición ──────────────────────────────────────────────────
            section_title("Composición del caso")
            _rc1, _rc2, _rc3, _rc4, _rc5 = st.columns(5)
            _rc1.metric("Total registros", f"{_n_total:,}")
            _rc2.metric("Normal real",     f"{_n_real_norm:,}" if _n_real_norm is not None else "—")
            _rc3.metric("Anormal real",    f"{_n_real_abn:,}"  if _n_real_abn  is not None else "—")
            _rc4.metric("Pred Normal",     f"{_n_pred_norm:,}")
            _rc5.metric("Pred Anormal",    f"{_n_pred_abn:,}")

            # ── Métricas por clase + globales (solo cuando hay CM) ────────────
            if _eval and _eval.get("cm") is not None:
                _cm = _eval["cm"]
                _tn, _fp_v, _fn, _tp = _cm.ravel() if _cm.size == 4 else (0, 0, 0, 0)

                def _safe_div(a, b):
                    return float(a) / float(b) if b else 0.0

                _rec_norm  = _safe_div(_tn, _tn + _fp_v)
                _prec_norm = _safe_div(_tn, _tn + _fn)
                _f1_norm   = _safe_div(2 * _prec_norm * _rec_norm, _prec_norm + _rec_norm)

                _rec_abn   = _safe_div(_tp, _tp + _fn)
                _prec_abn  = _safe_div(_tp, _tp + _fp_v)
                _f1_abn    = _safe_div(2 * _prec_abn * _rec_abn, _prec_abn + _rec_abn)

                _acc_m   = _safe_div(_tp + _tn, _n_total)
                _bal_acc = (_rec_norm + _rec_abn) / 2
                _f1_mac  = (_f1_norm + _f1_abn) / 2
                _n_errs  = int(_fp_v + _fn)

                st.write("")
                section_title("Métricas por clase")
                _rp1, _rp2, _rp3, _rp4, _rp5, _rp6 = st.columns(6)
                _rp1.metric("Recall Normal",     f"{_rec_norm:.1%}",
                            help="Especificidad: TN / (TN + FP)")
                _rp2.metric("Precision Normal",  f"{_prec_norm:.1%}",
                            help="TN / (TN + FN)")
                _rp3.metric("F1 Normal",         f"{_f1_norm:.1%}")
                _rp4.metric("Recall Anormal",    f"{_rec_abn:.1%}",
                            help="Sensibilidad: TP / (TP + FN)")
                _rp5.metric("Precision Anormal", f"{_prec_abn:.1%}",
                            help="TP / (TP + FP)")
                _rp6.metric("F1 Anormal",        f"{_f1_abn:.1%}")

                st.write("")
                section_title("Métricas globales")
                _rg1, _rg2, _rg3, _rg4 = st.columns(4)
                _rg1.metric("Accuracy",          f"{_acc_m:.1%}")
                _rg2.metric("Balanced Accuracy", f"{_bal_acc:.1%}")
                _rg3.metric("F1-macro",          f"{_f1_mac:.1%}")
                _rg4.metric("Errores totales",   f"{_n_errs:,}",
                            help=f"FP (Normal clasificado Anormal) = {_fp_v:,}  ·  "
                                 f"FN (Anormal clasificado Normal) = {_fn:,}")

                st.write("")
                section_title("Matriz de confusión binaria (caso actual)")
                _cm_df = pd.DataFrame(
                    _cm,
                    index=["Real: Normal", "Real: Anormal"],
                    columns=["Pred: Normal", "Pred: Anormal"],
                )
                with st.container(border=True):
                    st.dataframe(_cm_df, use_container_width=False)
                    st.caption(
                        f"TN={_tn:,}  FP={_fp_v:,}  FN={_fn:,}  TP={_tp:,} "
                        "· positivo = Anormal"
                    )

            _pred_vis_df = (
                _feat_df[["time_second", "rhythm_label"]].copy()
                if "time_second" in _feat_df.columns else pd.DataFrame()
            )
            if "beat_type" in _feat_df.columns:
                _pred_vis_df["beat_type"] = _feat_df["beat_type"].values
            _pred_vis_df["prediccion"] = _y_pred_bin

            # Referencia temporal: mínimo time_second absoluto de las predicciones.
            # El eje x del ECG va de 0 a max_seconds (relativo);
            # t_rel = time_second - _segment_start_abs garantiza que los vrects
            # caigan dentro del rango visible aunque time_second sea p. ej. 2021 s.
            _segment_start_abs = (
                float(_pred_vis_df["time_second"].min())
                if not _pred_vis_df.empty and "time_second" in _pred_vis_df.columns
                else 0.0
            )

            # Duración real del fragmento ECG disponible
            _duration_available = float(len(_disp_sig)) / TARGET_FS

            st.write("")
            section_title("ECG con regiones predichas")

            if _duration_available <= 0:
                callout("warn", "Señal no disponible",
                        "No hay muestras suficientes para graficar.")
            else:
                # Defaults de inicio y duración
                _cid_int = int(_active_cid) if _active_cid is not None else -1
                if _cid_int == 337:
                    _vstart_def   = min(589.0, max(0.0, _duration_available - 5.0))
                    _vdur_def     = min(80.0,  max(5.0, _duration_available - _vstart_def))
                    if _vdur_def < 5.0:
                        _vstart_def = max(0.0, _duration_available - 80.0)
                        _vdur_def   = min(80.0, _duration_available - _vstart_def)
                else:
                    _vstart_def = 0.0
                    _vdur_def   = min(120.0, _duration_available)

                _ecg_key_pfx = str(_active_cid) if _active_cid is not None else "uploaded"

                # Slider 1 — inicio relativo
                _max_start = max(0.0, _duration_available - 1.0)
                _sl_col1, _sl_col2 = st.columns(2)
                with _sl_col1:
                    _view_start = _safe_slider(
                        "Inicio relativo ECG (s)",
                        min_value=0.0,
                        max_value=_max_start,
                        value=_clamp(_vstart_def, 0.0, _max_start),
                        step=1.0,
                        key=f"ecg_view_start_{_ecg_key_pfx}",
                    )

                # Slider 2 — duración (rango depende del inicio elegido)
                _max_dur     = max(1.0, _duration_available - _view_start)
                _min_dur     = min(5.0, _max_dur)
                _vdur_clamped = _clamp(_vdur_def, _min_dur, _max_dur)
                with _sl_col2:
                    _view_duration = _safe_slider(
                        "Duración mostrada (s)",
                        min_value=_min_dur,
                        max_value=min(_max_dur, 300.0),
                        value=_vdur_clamped,
                        step=1.0,
                        key=f"ecg_view_duration_{_ecg_key_pfx}",
                    )

                # Recortar señal a la ventana elegida
                _start_samp = int(_view_start * TARGET_FS)
                _end_samp   = min(len(_disp_sig),
                                  int((_view_start + _view_duration) * TARGET_FS))
                _sig_view   = _disp_sig[_start_samp:_end_samp]

                # segment_start_time = tiempo absoluto VitalDB del inicio de la ventana
                _segment_start_time_plot = _segment_start_abs + _view_start

                if _view_start > 20:
                    st.caption(
                        f"Vista centrada en t_rel ≈ {int(_view_start)}–"
                        f"{int(_view_start + _view_duration)} s. "
                        "Ajusta los controles para explorar otras zonas."
                    )

                with st.container(border=True):
                    st.plotly_chart(
                        plot_ecg_with_binary_prediction_bands(
                            signal=_sig_view,
                            fs=TARGET_FS,
                            pred_df=_pred_vis_df,
                            time_col="time_second",
                            pred_col="prediccion",
                            segment_start_time=_segment_start_time_plot,
                            max_seconds=_view_duration,
                        ),
                        use_container_width=True,
                        config={"displayModeBar": True},
                    )
                    st.caption(
                        "Fondo rojo = regiones donde el modelo predice Anormal. "
                        "Línea gris = señal ECG visualizada. "
                        "Usa el zoom de Plotly para explorar regiones específicas."
                    )

            st.write("")
            section_title("Tabla: Real vs Predicción binaria")

            _show_cols_base = ["time_second", "rhythm_label"]
            if "beat_type" in _feat_df.columns:
                _show_cols_base.append("beat_type")
            _res_df = _feat_df[_show_cols_base].copy()
            _res_df["t_rel_s"]    = (_res_df["time_second"] - _segment_start_abs).round(2)
            _res_df["Real"]       = [_display(v) for v in (_y_true_bin if _y_true_bin is not None else ["—"] * len(_y_pred_bin))]
            _res_df["Predicción"] = [_display(v) for v in _y_pred_bin]
            if _y_true_bin is not None:
                _correct_mask = pd.Series(
                    np.asarray(_y_true_bin).astype(str) == np.asarray(_y_pred_bin).astype(str),
                    index=_res_df.index,
                )
                _res_df["✓/✗"] = _correct_mask.map({True: "✓", False: "✗"})
            if "time_second" in _res_df.columns:
                _res_df["time_second"] = _res_df["time_second"].round(3)

            _filter_opts = [
                "Todos",
                "Solo Normal real",
                "Solo Anormal real",
                "Solo pred Normal",
                "Solo pred Anormal",
                "Solo errores",
            ]
            _col_filt, _ = st.columns([3, 5])
            with _col_filt:
                _tbl_filter = st.selectbox(
                    "Filtrar tabla", _filter_opts, index=0, key="tbl_filter_sel"
                )

            _res_display = _res_df.copy()
            if _tbl_filter == "Solo Normal real":
                _res_display = _res_display[_res_display["Real"] == "Normal"]
            elif _tbl_filter == "Solo Anormal real":
                _res_display = _res_display[_res_display["Real"] == "Anormal"]
            elif _tbl_filter == "Solo pred Normal":
                _res_display = _res_display[_res_display["Predicción"] == "Normal"]
            elif _tbl_filter == "Solo pred Anormal":
                _res_display = _res_display[_res_display["Predicción"] == "Anormal"]
            elif _tbl_filter == "Solo errores" and "✓/✗" in _res_display.columns:
                _res_display = _res_display[_res_display["✓/✗"] == "✗"]

            with st.container(border=True):
                _show_n = min(len(_res_display), 500)
                st.dataframe(
                    _res_display.head(500),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "t_rel_s":      st.column_config.NumberColumn(
                            "tiempo relativo (s)", format="%.2f", width="small"),
                        "time_second":  st.column_config.NumberColumn(
                            "time_second (abs)", format="%.3f", width="small"),
                        "rhythm_label": st.column_config.TextColumn(
                            "rhythm_label (original)", width="medium"),
                        "Real":         st.column_config.TextColumn(
                            "Real (binario)", width="medium"),
                        "Predicción":   st.column_config.TextColumn(
                            "Predicción", width="medium"),
                        "✓/✗":          st.column_config.TextColumn(width="small"),
                    },
                )
                st.caption(
                    "rhythm_label es contexto — evaluación: Normal (N) vs Anormal (≠ N). "
                    f"Mostrando {_show_n:,} de {len(_res_display):,} filas "
                    f"({len(_res_df):,} totales)."
                )

            if _y_true_bin is not None and "✓/✗" in _res_df.columns:
                n_err = (_res_df["✓/✗"] == "✗").sum()
                if n_err > 0:
                    with st.expander(f"Errores de clasificación ({n_err:,})", expanded=False):
                        err_tbl = (
                            _res_df[_res_df["✓/✗"] == "✗"]
                            .groupby(["Real", "Predicción"])
                            .size()
                            .reset_index(name="count")
                            .sort_values("count", ascending=False)
                        )
                        st.dataframe(err_tbl, use_container_width=True, hide_index=True)

            callout(
                "warn",
                "Demo académica — no para uso clínico",
                "Resultados sobre un caso individual. "
                "Métricas oficiales: test_f1_macro=0.615, accuracy=0.633.",
            )

# ===========================================================================
# ECG sin anotaciones — solo visualización
# ===========================================================================
else:
    st.html('<hr style="border-color:var(--line-1);margin:24px 0 16px">')
    callout(
        "warn",
        "Predicción no disponible para esta señal",
        "No fue posible ejecutar el modelo sobre este archivo porque el modelo actual "
        "requiere features tabulares compatibles con el pipeline de entrenamiento: "
        "intervalos RR, posición temporal y metadata clínica. "
        "La señal ECG se muestra únicamente como visualización. "
        "<br><br>"
        "En una versión futura, un modelo entrenado directamente sobre ECG crudo "
        "permitiría predecir Normal/Anormal a partir del <code>.npy</code> sin "
        "requerir anotaciones previas. "
        "<br>"
        "<span style='color:var(--fg-3);font-size:11.5px'>"
        "Esta funcionalidad es demostrativa y académica. "
        "No debe usarse para diagnóstico ni interpretación clínica real."
        "</span>",
    )

page_footer()
