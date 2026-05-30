"""Página 07 — Probar ECG.

Evalúa casos VitalDB conocidos desde archivos .npy, mostrando preprocesamiento
de señal ECG, predicciones del modelo tabular y comparación con etiquetas reales.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.paths import PROJECT_ROOT, DATA_DIR
from components.badges import badge
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.layout import page_header
from utils.loaders import load_model, load_model_metadata

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Demo case catalogue (precomputed metadata)
# ---------------------------------------------------------------------------
_DEMO_CASES: list[dict] = [
    {
        "case_id":    12,
        "tag":        "N + Ectopia ventricular",
        "description":"Ritmo sinusal mayoritario con latidos ventriculares ectópicos. Modelo acierta ~89%.",
        "classes":    ["N", "Patterned Ventricular Ectopy"],
        "n_beats":    1106,
        "accuracy":   0.889,
        "accent":     "teal",
        "icon":       "🫀",
    },
    {
        "case_id":    1314,
        "tag":        "N puro — modelo falla",
        "description":"Solo ritmo sinusal, pero el modelo tiene baja accuracy (8%). Ilustra limitaciones con NSR.",
        "classes":    ["N"],
        "n_beats":    1935,
        "accuracy":   0.084,
        "accent":     "err",
        "icon":       "⚠",
    },
    {
        "case_id":    1738,
        "tag":        "AFIB/AFL — detección perfecta",
        "description":"Fibrilación/flutter auricular sostenido. El modelo detecta el 100% de los latidos.",
        "classes":    ["AFIB/AFL"],
        "n_beats":    4712,
        "accuracy":   1.0,
        "accent":     "ok",
        "icon":       "✓",
    },
    {
        "case_id":    3519,
        "tag":        "Multi-arritmia: AFIB + VT + SVTA",
        "description":"Cuatro tipos de ritmo incluyendo VT. Alta accuracy (97%) con clases difíciles.",
        "classes":    ["AFIB/AFL", "SVTA", "VT", "Unclassifiable"],
        "n_beats":    2391,
        "accuracy":   0.971,
        "accent":     "blue",
        "icon":       "📊",
    },
    {
        "case_id":    2852,
        "tag":        "5 clases — caso muy difícil",
        "description":"Cinco tipos de ritmo distintos. Accuracy del 3%: el caso más desafiante del set.",
        "classes":    ["N", "Patterned Atrial Ectopy", "Patterned Ventricular Ectopy", "SND", "SVTA"],
        "n_beats":    659,
        "accuracy":   0.029,
        "accent":     "warn",
        "icon":       "⚡",
    },
]

_DEMO_FRAG_DIR = DATA_DIR / "demo" / "npy_cases"
_FEAT_COLS: list[str] = [
    "time_second", "analyzed_duration_sec", "total_beats", "caseend", "anestart",
    "aneend", "opstart", "opend", "height", "weight", "bmi", "preop_plt", "preop_pt",
    "preop_k", "preop_alb", "preop_ast", "preop_alt", "preop_cr", "tubesize",
    "intraop_uo", "intraop_crystalloid", "intraop_rocu", "rr_prev", "rr_next",
    "hr_inst_from_rr_prev", "position_in_case", "optype", "iv1", "aline1", "cline1",
]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
page_header(
    "Probar ECG",
    "Evalúa casos VitalDB conocidos o visualiza el preprocesamiento de señales ECG.",
    badge_html=badge("Demo interactiva", "info"),
)

# ---------------------------------------------------------------------------
# Callout metodológico
# ---------------------------------------------------------------------------
callout(
    "info",
    "Cómo funciona esta demo",
    "El modelo trabaja con <b>features tabulares</b> (RR intervals + metadatos clínicos). "
    "Para casos VitalDB conocidos (<code>case_XXXX.npy</code>), la app asocia el archivo con sus "
    "anotaciones reales y calcula predicciones comparables. "
    "Para archivos externos sin anotaciones ni metadata, solo es posible "
    "visualizar la señal y su preprocesamiento — no se calcula predicción ni evaluación. "
    "<br><span style='color:var(--fg-3)'>La señal ECG no alimenta el modelo directamente: "
    "se usa únicamente para visualización.</span>",
)

st.write("")

# ---------------------------------------------------------------------------
# Section 1: Demo cases grid
# ---------------------------------------------------------------------------
section_title("Casos VitalDB demo")

st.html(
    '<p style="font-size:12px;color:var(--fg-3);margin:0 0 12px">'
    'Haz clic en un caso para cargarlo automáticamente.'
    '</p>'
)

_cols = st.columns(5)
for _i, _dc in enumerate(_DEMO_CASES):
    with _cols[_i]:
        _is_active = st.session_state.get("_p7_active_case") == _dc["case_id"]
        _acc_color = (
            "var(--ok)"   if _dc["accuracy"] >= 0.80 else
            "var(--teal)" if _dc["accuracy"] >= 0.50 else
            "var(--warn)"  if _dc["accuracy"] >= 0.20 else
            "var(--err)"
        )
        _border = "2px solid var(--teal)" if _is_active else "1px solid var(--line-1)"
        st.html(
            f'<div style="border:{_border};border-radius:8px;padding:10px 10px 8px;'
            f'background:var(--bg-1);min-height:130px">'
            f'<div style="font-size:18px;margin-bottom:4px">{_dc["icon"]}</div>'
            f'<div style="font-size:11px;font-weight:600;color:var(--fg-0);'
            f'line-height:1.3;margin-bottom:6px">{_dc["tag"]}</div>'
            f'<div style="font-size:10px;color:var(--fg-3);margin-bottom:6px;line-height:1.4">'
            f'{_dc["description"][:60]}…</div>'
            f'<div style="font-size:10px;font-family:var(--mono);color:var(--fg-3)">'
            f'case_id {_dc["case_id"]} · {_dc["n_beats"]:,} latidos</div>'
            f'<div style="font-size:11px;font-weight:600;color:{_acc_color};'
            f'font-family:var(--mono);margin-top:4px">acc {_dc["accuracy"]:.0%}</div>'
            f'</div>'
        )
        if st.button(f"Cargar", key=f"load_case_{_dc['case_id']}", use_container_width=True):
            st.session_state["_p7_active_case"]   = _dc["case_id"]
            st.session_state["_p7_signal"]         = None
            st.session_state["_p7_proc_signal"]    = None
            st.session_state["_p7_feat_df"]        = None
            st.session_state["_p7_predictions"]    = None
            st.session_state["_p7_source"]         = "demo"
            st.rerun()

st.write("")

# ---------------------------------------------------------------------------
# Section 2: Download demo fragment
# ---------------------------------------------------------------------------
section_title("Descargar caso demo")

_active_cid = st.session_state.get("_p7_active_case", 12)
_dc_info = next((d for d in _DEMO_CASES if d["case_id"] == _active_cid), _DEMO_CASES[0])
_frag_path = _DEMO_FRAG_DIR / f"case_{_active_cid}.npy"

col_dl_a, col_dl_b, _ = st.columns([2, 2, 4])
with col_dl_a:
    if _frag_path.exists():
        _frag_bytes = _frag_path.read_bytes()
        st.download_button(
            label=f"⬇ Descargar case_{_active_cid}.npy (demo, 60 s)",
            data=_frag_bytes,
            file_name=f"case_{_active_cid}.npy",
            mime="application/octet-stream",
            use_container_width=True,
            help="Fragmento de 60 s del segmento válido de la señal. "
                 "Súbelo en la sección siguiente para ver el flujo completo.",
        )
    else:
        st.caption("Fragmento demo no disponible.")
with col_dl_b:
    st.caption(
        f"**{_dc_info['tag']}** · {_dc_info['n_beats']:,} latidos · "
        f"acc {_dc_info['accuracy']:.0%}"
    )

st.write("")

# ---------------------------------------------------------------------------
# Section 3: Upload .npy
# ---------------------------------------------------------------------------
section_title("Subir archivo ECG (.npy)")

st.html(
    '<p style="font-size:12px;color:var(--fg-3);margin:0 0 10px">'
    'Acepta cualquier archivo .npy 1D con señal ECG a 500 Hz. '
    'Para evaluación completa, usa un archivo con nombre <code>case_XXXX.npy</code>.'
    '</p>'
)
_uploaded = st.file_uploader(
    "Seleccionar archivo .npy",
    type=["npy"],
    key="npy_uploader",
    label_visibility="collapsed",
)

if _uploaded is not None:
    # New upload overrides demo selection
    _upload_cid_detected = None
    try:
        from utils.case_eval import extract_case_id_from_filename
        _upload_cid_detected = extract_case_id_from_filename(_uploaded.name)
    except Exception:
        pass
    if st.session_state.get("_p7_last_upload") != _uploaded.name:
        st.session_state["_p7_last_upload"]   = _uploaded.name
        st.session_state["_p7_active_case"]   = _upload_cid_detected
        st.session_state["_p7_signal"]        = None
        st.session_state["_p7_proc_signal"]   = None
        st.session_state["_p7_feat_df"]       = None
        st.session_state["_p7_predictions"]   = None
        st.session_state["_p7_source"]        = "upload"
        st.session_state["_p7_upload_file"]   = _uploaded
        st.rerun()

# ---------------------------------------------------------------------------
# Resolve active case & signal
# ---------------------------------------------------------------------------
try:
    from utils.case_eval import (
        find_annotation_file, load_case_annotations, load_case_metadata,
        get_case_features_from_parquet, build_tabular_features_from_case,
        load_npy_signal, summarize_npy_signal, find_valid_segments,
        predict_case_windows, evaluate_case_predictions, TARGET_FS, WAVEFORMS_DIR,
    )
    from components.ecg_viewer import (
        plot_ecg_signal, plot_raw_vs_processed,
        plot_annotations_on_ecg, plot_ecg_with_prediction_regions,
    )
    _EVAL_MODULES_OK = True
except ImportError as _import_err:
    _EVAL_MODULES_OK = False
    callout(
        "err",
        "Módulos de evaluación no disponibles",
        f"Falta una dependencia requerida: <code>{_import_err}</code>. "
        "Verifica que <code>requirements.txt</code> incluya plotly, numpy y pandas.",
    )
    st.stop()

_active_cid = st.session_state.get("_p7_active_case")
_source     = st.session_state.get("_p7_source", "demo")

# Load signal if not yet loaded
if _active_cid is not None and st.session_state.get("_p7_signal") is None:
    _sig_path = None
    if _source == "demo":
        # Use full waveform if available, else fragment
        _full_path = WAVEFORMS_DIR / f"case_{_active_cid}.npy"
        _frag_p    = _DEMO_FRAG_DIR / f"case_{_active_cid}.npy"
        _sig_path  = _full_path if _full_path.exists() else _frag_p
    elif _source == "upload":
        _upload_obj = st.session_state.get("_p7_upload_file")
        if _upload_obj is not None:
            _upload_obj.seek(0)
            _sig_path = _upload_obj

    if _sig_path is not None:
        try:
            st.session_state["_p7_signal"] = load_npy_signal(_sig_path)
        except Exception as _load_err:
            callout("err", "Error al cargar el archivo .npy", str(_load_err))

_signal = st.session_state.get("_p7_signal")

if _signal is None and _active_cid is None:
    st.html(
        '<div class="placeholder-block" style="min-height:120px;padding:24px">'
        '<div class="ph-mono">sin caso activo</div>'
        '<div class="ph-title">Selecciona un caso demo o sube un archivo .npy</div>'
        '</div>'
    )
    st.stop()

if _signal is None:
    st.warning("No se pudo cargar la señal. Verifica el archivo o selecciona un caso demo.")
    st.stop()

# ---------------------------------------------------------------------------
# Determine Caso A vs Caso B
# ---------------------------------------------------------------------------
_ann_path = find_annotation_file(_active_cid) if _active_cid else None
_is_case_a = (_active_cid is not None) and (_ann_path is not None)

st.write("")
st.html('<hr style="border-color:var(--line-1);margin:4px 0 20px">')

# Case badge
if _is_case_a:
    st.html(
        f'<div style="display:inline-flex;align-items:center;gap:10px;padding:6px 14px;'
        f'background:rgba(45,212,191,0.08);border:1px solid rgba(45,212,191,0.3);'
        f'border-radius:6px;margin-bottom:14px">'
        f'<span style="color:#2dd4bf;font-weight:600;font-size:13px">✓ Caso VitalDB reconocido</span>'
        f'<span style="color:#8a98b5;font-size:11px;font-family:var(--mono)">case_id = {_active_cid}</span>'
        f'</div>'
    )
else:
    st.html(
        '<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;'
        'background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);'
        'border-radius:6px;margin-bottom:14px">'
        '<span style="color:#fbbf24;font-weight:600;font-size:13px">⚠ Archivo sin anotaciones</span>'
        '<span style="color:#8a98b5;font-size:11px">predicción tabular no disponible</span>'
        '</div>'
    )

# ---------------------------------------------------------------------------
# Section 4: Signal summary
# ---------------------------------------------------------------------------
section_title("Resumen de la señal")

_summary = summarize_npy_signal(_signal, fs=TARGET_FS)
_segs    = find_valid_segments(_signal, fs=TARGET_FS, min_duration_s=5.0, max_segments=5)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Muestras", f"{_summary['n_samples']:,}")
c2.metric("Duración (s)", f"{_summary['duration_s']:,.0f}")
c3.metric("fs asumida", f"{int(TARGET_FS)} Hz")
c4.metric("NaN", f"{_summary['nan_pct']:.1f}%")
c5.metric("Segmentos válidos", str(len(_segs)))

# Choose best display segment
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

_fname_label = f"case_{_active_cid}.npy" if _active_cid else "archivo .npy"
st.plotly_chart(
    plot_ecg_signal(
        _disp_sig, TARGET_FS,
        title=f"ECG crudo · {_fname_label} · primeros 10 s del segmento válido",
        start_time_offset=_disp_offset,
        max_seconds=10.0,
    ),
    use_container_width=True,
    config={"displayModeBar": False},
)

# ---------------------------------------------------------------------------
# Section 6: ECG preprocesado
# ---------------------------------------------------------------------------
section_title("Preprocesamiento ECG")

_proc_sig = st.session_state.get("_p7_proc_signal")
_proc_err = st.session_state.get("_p7_proc_error")

if st.button("Preprocesar señal (filtro 0.5–40 Hz · normalización)", key="btn_preproc"):
    with st.spinner("Procesando…"):
        try:
            from src.pipeline import preprocess_ecg
            st.session_state["_p7_proc_signal"] = preprocess_ecg(
                _disp_sig, original_fs=TARGET_FS
            )
            st.session_state["_p7_proc_error"] = None
        except Exception as _pe:
            st.session_state["_p7_proc_error"]  = str(_pe)
            st.session_state["_p7_proc_signal"] = None

_proc_sig = st.session_state.get("_p7_proc_signal")
_proc_err = st.session_state.get("_p7_proc_error")

if _proc_err:
    st.error(_proc_err)
elif _proc_sig is not None:
    _fr, _fp = plot_raw_vs_processed(
        _disp_sig, _proc_sig,
        raw_fs=TARGET_FS,
        start_time_offset=_disp_offset,
        max_seconds=10.0,
    )
    col_r, col_p = st.columns(2)
    with col_r:
        st.plotly_chart(_fr, use_container_width=True, config={"displayModeBar": False})
    with col_p:
        st.plotly_chart(_fp, use_container_width=True, config={"displayModeBar": False})

# ===========================================================================
# CASO A — prediction + evaluation
# ===========================================================================
if _is_case_a:
    st.html('<hr style="border-color:var(--line-1);margin:24px 0 16px">')
    section_title(f"Predicción y evaluación · case_id = {_active_cid}")

    _ann      = load_case_annotations(_ann_path)
    _meta_row = load_case_metadata(_active_cid)
    _model_p  = load_model()
    _meta_md  = load_model_metadata()

    _feat_cols_a = (
        (_meta_md.get("numeric_features", []) + _meta_md.get("categorical_features", []))
        if _meta_md else _FEAT_COLS
    )

    # Annotation info
    col_ann1, col_ann2, col_ann3 = st.columns(3)
    col_ann1.metric("Latidos anotados (válidos)", f"{len(_ann):,}")
    col_ann2.metric("Clases reales",
                    str(_ann["rhythm_label"].nunique()) if "rhythm_label" in _ann.columns else "—")
    col_ann3.metric("Metadata disponible", "Sí" if _meta_row is not None else "No")

    # ECG with annotation markers
    if len(_ann) > 0:
        _ann_win = _ann[
            (_ann["time_second"] >= _disp_offset) &
            (_ann["time_second"] <= _disp_offset + 10)
        ]
        if len(_ann_win) > 0:
            with st.container(border=True):
                card_header(
                    "ECG con latidos anotados",
                    f"{len(_ann_win)} latidos en la ventana de 10 s · colores por rhythm_label",
                )
                st.plotly_chart(
                    plot_annotations_on_ecg(
                        _disp_sig, _ann_win, fs=TARGET_FS,
                        start_time_offset=_disp_offset, max_seconds=10.0,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # Load/compute predictions
    if _model_p is None:
        callout("warn", "Modelo no disponible",
                "No se encontró <code>models/tabular_best_model_pipeline.joblib</code>.")
    elif len(_ann) == 0:
        callout("warn", "Sin anotaciones válidas",
                "El archivo de anotaciones no tiene filas con rhythm_label válido.")
    else:
        # Fast path: pre-computed parquet rows (guarantees feature consistency)
        _feat_df = st.session_state.get("_p7_feat_df")
        if _feat_df is None:
            _feat_df = get_case_features_from_parquet(_active_cid, _feat_cols_a)
            _feat_src = "parquet pre-calculado"
            if _feat_df is None:
                # Fallback: rebuild from annotations + metadata
                if _meta_row is None:
                    callout(
                        "warn", "Metadata no disponible",
                        f"No se encontró case_id={_active_cid} en metadata.csv ni en el parquet. "
                        "Sin las 25 features clínicas requeridas no es posible predecir.",
                    )
                    _feat_df = None
                else:
                    _feat_df, _missing = build_tabular_features_from_case(
                        _ann, _meta_row, _feat_cols_a
                    )
                    _feat_src = "reconstruido desde anotaciones + metadata"
                    if _missing:
                        callout(
                            "warn",
                            f"{len(_missing)} features faltantes",
                            "El pipeline imputará con medianas: "
                            + ", ".join(f"<code>{f}</code>" for f in _missing),
                        )
            st.session_state["_p7_feat_df"] = _feat_df

        if _feat_df is not None and len(_feat_df) > 0:

            # Compute predictions
            _preds = st.session_state.get("_p7_predictions")
            if _preds is None:
                try:
                    _preds = predict_case_windows(_feat_df, _model_p, _feat_cols_a)
                    st.session_state["_p7_predictions"] = _preds
                except Exception as _pe:
                    callout("err", "Error en predicción", str(_pe))
                    _preds = None

            if _preds is not None:
                _y_true_a = (
                    _feat_df["rhythm_label"].to_numpy()
                    if "rhythm_label" in _feat_df.columns else None
                )
                _eval = evaluate_case_predictions(_y_true_a, _preds) if _y_true_a is not None else None

                if _eval and _eval.get("warning"):
                    callout("warn", "Advertencia de predicción", _eval["warning"])

                # KPIs
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Latidos evaluados",
                          f"{_eval['n_beats']:,}" if _eval else f"{len(_preds):,}")
                e2.metric("Aciertos",
                          f"{_eval['n_correct']:,}" if _eval else "—")
                e3.metric("Accuracy del caso",
                          f"{_eval['accuracy']:.1%}" if _eval and _eval['accuracy'] is not None else "—")
                e4.metric("Clases presentes",
                          str(len(_eval['classes_present'])) if _eval else "—")

                # Build predictions DataFrame for visualization
                _pred_vis_df = _feat_df[["time_second", "rhythm_label"]].copy() if "time_second" in _feat_df.columns else pd.DataFrame()
                if "beat_type" in _feat_df.columns:
                    _pred_vis_df["beat_type"] = _feat_df["beat_type"].values
                _pred_vis_df["prediccion"] = _preds

                # ECG with prediction regions
                st.write("")
                section_title("ECG con regiones predichas")
                _max_ecg_sec = st.slider(
                    "Ventana ECG (segundos)", min_value=10, max_value=60, value=30,
                    step=5, key="ecg_window_slider",
                )
                with st.container(border=True):
                    st.plotly_chart(
                        plot_ecg_with_prediction_regions(
                            _disp_sig, TARGET_FS,
                            _pred_vis_df,
                            time_col="time_second",
                            pred_col="prediccion",
                            real_col="rhythm_label",
                            beat_type_col="beat_type",
                            max_seconds=float(_max_ecg_sec),
                            start_offset=_disp_offset,
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.caption(
                        "Regiones coloreadas por ritmo predicho. "
                        "Triángulos rojos = predicción incorrecta (real ≠ predicho)."
                    )

                # Comparison table
                st.write("")
                section_title("Tabla de comparación: real vs predicción")
                _show_cols = ["time_second", "rhythm_label"]
                if "beat_type" in _feat_df.columns:
                    _show_cols.append("beat_type")
                _res_df = _feat_df[_show_cols].copy()
                _res_df["prediccion"] = _preds
                if _y_true_a is not None:
                    _res_df["correcto"] = (
                        pd.Series(np.asarray(_y_true_a).astype(str))
                        == pd.Series(np.asarray(_preds).astype(str))
                    ).map({True: "✓", False: "✗"}).values
                _res_df["time_second"] = _res_df["time_second"].round(3)

                col_filt, _ = st.columns([2, 6])
                with col_filt:
                    _show_all = st.checkbox("Mostrar solo errores", value=False)
                _res_display = _res_df.loc[_res_df["correcto"] == "✗"] if _show_all else _res_df

                with st.container(border=True):
                    st.dataframe(
                        _res_display.head(300),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "correcto": st.column_config.TextColumn(width="small"),
                            "time_second": st.column_config.NumberColumn(
                                "time (s)", format="%.3f", width="small"
                            ),
                        },
                    )
                    st.caption(
                        "beat_type es descriptivo únicamente — no entra al modelo. "
                        f"Mostrando {min(len(_res_display), 300):,} de {len(_res_display):,} filas."
                    )

                # Top errors
                if _eval and _eval["error_table"] is not None and len(_eval["error_table"]) > 0:
                    with st.expander("Principales errores de clasificación", expanded=False):
                        st.dataframe(
                            _eval["error_table"],
                            use_container_width=True, hide_index=True,
                        )

                callout(
                    "warn",
                    "Demo académica — no para uso clínico",
                    "Resultados exploratorios. El modelo fue entrenado con otros casos; "
                    "la accuracy reportada aquí no coincide necesariamente con las métricas oficiales.",
                )

# ===========================================================================
# CASO B — solo visualización
# ===========================================================================
else:
    st.html('<hr style="border-color:var(--line-1);margin:24px 0 16px">')
    callout(
        "warn",
        "No se encontraron anotaciones reales para este archivo",
        "El modelo tabular requiere <b>30 features</b>: "
        "5 derivadas de anotaciones de latidos (RR intervals, posición) "
        "y 25 de metadata clínica del caso. "
        "Sin estas fuentes no es posible construir el vector de features, "
        "por lo que esta carga se limita a visualización y preprocesamiento. "
        "<br><br>"
        "Para evaluación completa, el archivo debe llamarse <code>case_XXXX.npy</code> "
        "y existir el <code>Annotation_file_XXXX.csv</code> correspondiente.",
    )
