"""Página 01 — Inicio · landing dashboard con datos reales."""

import streamlit as st
import pandas as pd

from components.badges  import badge, badge_row, pill
from components.cards   import callout, card_header, kv_table, metric_card, section_title
from components.charts  import bar_chart_h, mini_ecg_placeholder
from components.layout import page_footer
from utils.loaders      import (
    load_classification_report,
    load_model_metadata,
    load_model_final_official,
    load_model_comparison_history,
    load_train_test_split_summary,
    load_class_support_train_test,
)

# ── Global setup ──────────────────────────────────────────────────────────────
meta         = load_model_metadata()
df_final     = load_model_final_official()
df_hist      = load_model_comparison_history()
df_cls       = load_classification_report()
split_info   = load_train_test_split_summary()
df_support   = load_class_support_train_test()

# Modelo final: preferir tabular_model_final_official.csv, luego metadata
if df_final is not None and not df_final.empty:
    _row      = df_final.iloc[0]
    winner_raw  = str(_row.get("model", "linear_svc"))
    f1_val      = float(_row["test_f1_macro"])   if "test_f1_macro"          in _row.index else None
    acc_val     = float(_row["test_accuracy"])    if "test_accuracy"          in _row.index else None
    prec_val    = float(_row["test_precision_macro"]) if "test_precision_macro"   in _row.index else None
    rec_val     = float(_row["test_recall_macro"])    if "test_recall_macro"       in _row.index else None
    bal_val     = float(_row["test_balanced_accuracy"]) if "test_balanced_accuracy" in _row.index else None
else:
    winner_raw  = meta.get("winner_model", "linear_svc") if meta else "linear_svc"
    wm          = (meta or {}).get("winner_metrics", {})
    f1_val      = wm.get("test_f1_macro")
    acc_val     = wm.get("test_accuracy")
    prec_val    = wm.get("test_precision_macro")
    rec_val     = wm.get("test_recall_macro")
    bal_val     = wm.get("test_balanced_accuracy")

winner_nice = winner_raw.replace("_", " ").title()
f1_str      = f"{f1_val:.3f}" if f1_val is not None else "—"

# Dataset counts — CSV first, then metadata, then official hardcoded fallbacks
def _int_val(d: dict | None, key: str, fallback: int) -> int:
    v = (d or {}).get(key)
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return fallback

n_train_grp = _int_val(split_info, "n_train_groups", 385)
n_test_grp  = _int_val(split_info, "n_test_groups",  97)
train_rows  = _int_val(split_info, "n_train_rows",   510287)
test_rows   = _int_val(split_info, "n_test_rows",    129173)
total_cases = n_train_grp + n_test_grp   # 482
total_rows  = train_rows  + test_rows    # 639 460

# Class-level support from CSV
def _support(df: "pd.DataFrame | None", cls: str, split: str) -> int | None:
    if df is None or "class" not in df.columns or split not in df.columns:
        return None
    row = df[df["class"] == cls]
    if row.empty:
        return None
    v = row[split].iloc[0]
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None

n_normal_total    = _support(df_support, "normal",   "total")   or 392623
n_abnormal_total  = _support(df_support, "abnormal", "total")   or 246837
n_normal_train    = _support(df_support, "normal",   "train")   or 313712
n_abnormal_train  = _support(df_support, "abnormal", "train")   or 196575
n_normal_test     = _support(df_support, "normal",   "test")    or 78911
n_abnormal_test   = _support(df_support, "abnormal", "test")    or 50262

# Features: 57 numéricas + 16 categóricas = 73 originales → 162 tras OHE
n_num_feat  = len(meta.get("numeric_features", []))     if meta else 57
n_cat_feat  = len(meta.get("categorical_features", [])) if meta else 16
n_features_orig = n_num_feat + n_cat_feat

# Task is binary: normal (N) vs abnormal (not N)
n_classes = 2

# Count candidate models from exploratory history (not from final test CSV)
if df_hist is not None:
    df_exp    = df_hist[df_hist["run_type"] == "exploratory_150_cases"] if "run_type" in df_hist.columns else df_hist
    n_models  = len(df_exp)
else:
    n_models  = 5

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="hero-block">
          <div style="display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap">
            {badge("VitalDB · 2026","muted")}
            {badge("Machine Learning","info")}
            {badge("Demo académica","warn")}
          </div>
          <h1>Detección de
              <em>arritmias intraoperatorias</em>
              con señales ECG y modelos ML</h1>
          <p>
            Esta demo recorre el pipeline completo: anotaciones de ritmo por latido
            de la VitalDB Arrhythmia Database, construcción de features tabulares
            (metadatos clínicos + RR intervals) y clasificación binaria
            <b>normal vs anormal</b> usando modelos clásicos de aprendizaje automático.
          </p>
          <div class="hero-tags">
            {badge("scikit-learn","muted")}
            {badge("Binario: normal / anormal","muted")}
            {badge(f"{n_features_orig} features originales","muted")}
            {badge("GroupSplit (case_id)","muted")}
            {badge("Demo académica","warn")}
          </div>
          <div class="hero-foot">
            <span>VitalDB Arrhythmia Database</span>
            <span class="sep">·</span>
            <span>v1.0 pipeline</span>
            <span class="sep">·</span>
            <span>Entrenado: {meta.get("training_datetime","—")[:10] if meta else "2026-05-30"}</span>
            <div class="hero-status">
              <span class="led"></span>
              Pipeline OK · datos cargados
            </div>
          </div>
        </div>""",
    unsafe_allow_html=True,
)

# ── ECG decorativo ────────────────────────────────────────────────────────────
ecg_fig = mini_ecg_placeholder(height=110, n_beats=7)
st.plotly_chart(ecg_fig, use_container_width=True, config={"displayModeBar": False})

# ── MÉTRICAS PRINCIPALES ──────────────────────────────────────────────────────
section_title("Resumen del proyecto")

_metric_cols = []
_metric_cols.append(("Casos quirúrgicos", str(total_cases),
                     f"{n_train_grp} train · {n_test_grp} test", "blue"))
_metric_cols.append(("Latidos anotados",
                     f"{total_rows:,}".replace(",", " "),
                     f"train {train_rows:,} · test {test_rows:,}".replace(",", " "),
                     "teal"))
_metric_cols.append(("Tarea", "Binario", "normal vs anormal", "warn"))
_metric_cols.append(("Candidatos evaluados", str(n_models),
                     "benchmark exploratorio · 150 casos", "blue"))
if f1_val is not None:
    _metric_cols.append(("F1-macro (test)", f1_str,
                         f"Linear SVC · dataset completo", "teal"))

_mc = st.columns(len(_metric_cols))
for _col, (_label, _val, _helper, _accent) in zip(_mc, _metric_cols):
    with _col:
        metric_card(_label, _val, helper=_helper, accent=_accent, helper_kind="muted")

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ── CARDS PRINCIPALES ─────────────────────────────────────────────────────────
section_title("Resultados")

col_a, col_b, col_c = st.columns([1.1, 1, 1])

# Card A — Modelo final oficial
with col_a:
    with st.container(border=True):
        card_header(
            "Modelo final · desplegado",
            "dataset completo · corrida definitiva",
            right_html=(badge("winner", "winner") + "&nbsp;" + badge("full dataset", "ok")),
        )

        st.markdown(
            f"""<div style="display:flex;align-items:baseline;gap:14px;margin:6px 0 12px">
                  <div style="font-size:28px;font-weight:600;color:var(--fg-0);
                              letter-spacing:-.01em;">{winner_nice}</div>
                  <span style="font-family:var(--mono);font-size:11px;
                               color:var(--fg-3)">scikit-learn</span>
                </div>""",
            unsafe_allow_html=True,
        )

        kv_table([
            ("F1-macro",        f'<b style="color:var(--teal)">{f1_str}</b>'),
            ("Balanced acc.",   f"{bal_val:.3f}"  if bal_val  is not None else "—"),
            ("Precision",       f"{prec_val:.3f}" if prec_val is not None else "—"),
            ("Recall",          f"{rec_val:.3f}"  if rec_val  is not None else "—"),
            ("Accuracy",        f"{acc_val:.3f}"  if acc_val  is not None else "—"),
            ("Features orig.",  str(n_features_orig)),
        ])

        if meta and "best_hyperparams_per_model" in meta:
            hp = meta["best_hyperparams_per_model"].get(winner_raw, {})
            if hp:
                st.caption("Mejores hiperparámetros:")
                def _fv(v):
                    try:
                        if isinstance(v, bool): return str(v)
                        if isinstance(v, float): return f"{v:.4g}"
                        if isinstance(v, int): return str(v)
                    except Exception: pass
                    return str(v)
                hp_str = " · ".join(
                    f"{k.replace('clf__','').replace('preprocessor__','')}={_fv(v)}"
                    for k, v in hp.items()
                )
                st.code(hp_str, language=None)

# Card B — Dataset
with col_b:
    with st.container(border=True):
        card_header("Resumen del dataset", "VitalDB")
        kv_table([
            ("Casos totales",   str(total_cases)),
            ("Registros",       f"{total_rows:,}".replace(",", " ")),
            ("Tipo de datos",   "Anotaciones por latido + metadatos clínicos"),
            ("Train registros", f"{train_rows:,}".replace(",", " ")),
            ("Test registros",  f"{test_rows:,}".replace(",", " ")),
            ("Tarea",           "Binario: normal (N) vs anormal (no N)"),
            ("Features orig.",  f"{n_features_orig} ({n_num_feat} num + {n_cat_feat} cat)"),
            ("Features OHE",    "162 tras One-Hot Encoding"),
        ])
        st.markdown(
            f'<div style="margin-top:10px;line-height:2">'
            f'{badge("normal", "ok")} {badge(f"{n_normal_total:,}".replace(",", " "), "ok")}'
            f'&nbsp;&nbsp;'
            f'{badge("anormal", "err")} {badge(f"{n_abnormal_total:,}".replace(",", " "), "err")}'
            f'</div>',
            unsafe_allow_html=True,
        )

# Card C — Benchmark exploratorio
with col_c:
    with st.container(border=True):
        card_header(
            "Benchmark exploratorio",
            "5 candidatos · 150 casos · F1-macro",
        )

        if df_hist is not None and "test_f1_macro" in df_hist.columns:
            df_exp_chart = (
                df_hist[df_hist["run_type"] == "exploratory_150_cases"].copy()
                if "run_type" in df_hist.columns else df_hist.copy()
            )
            df_exp_chart = df_exp_chart.sort_values("test_f1_macro", ascending=False)
            labels = df_exp_chart["model"].str.replace("_", " ").str.title().tolist()
            values = df_exp_chart["test_f1_macro"].tolist()
            fig = bar_chart_h(
                labels=labels,
                values=values,
                color="#4a8cff",
                accent_top=True,
                value_fmt=".3f",
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "Corrida exploratoria · 150 casos · no representa el rendimiento final. "
                "Modelo desplegado: Linear SVC (corrida con dataset completo)."
            )
        else:
            kv_table([
                ("Modelo desplegado",  winner_nice),
                ("F1-macro test",      f1_str),
                ("Candidatos",         str(n_models)),
            ])
            st.caption("Corrida exploratoria: benchmark rápido de candidatos.")

# ── AVISO ──────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
callout(
    "warn",
    "Aviso · Demo académica",
    "Esta aplicación no permite cargar datos nuevos ni realizar diagnóstico clínico. "
    "Los resultados provienen de un pipeline ya ejecutado sobre la VitalDB Arrhythmia Database. "
    "<b>No usar para decisiones médicas reales.</b>",
)

# ── DESCRIPCIÓN DEL PROYECTO ──────────────────────────────────────────────────
section_title("Descripción del proyecto")

col_d, col_e = st.columns(2)

with col_d:
    with st.container(border=True):
        card_header("Problema clínico", "contexto")
        st.markdown(
            """<div class="kv-table" style="row-gap:10px">
              <div class="kv-key">¿Qué se clasifica?</div>
              <div class="kv-val">Ritmos cardíacos intraoperatorios como <b>normal</b> (ritmo sinusal, N) o <b>anormal</b> (cualquier arritmia)</div>
              <div class="kv-key">¿Por qué importa?</div>
              <div class="kv-val">Las arritmias intraoperatorias aumentan morbilidad y mortalidad si no se detectan a tiempo</div>
              <div class="kv-key">¿Qué hace el modelo?</div>
              <div class="kv-val">Clasifica cada latido anotado como <b>normal</b> o <b>anormal</b> usando features tabulares (metadatos clínicos + RR intervals)</div>
              <div class="kv-key">Target binario</div>
              <div class="kv-val"><code>normal = rhythm_label == "N"</code><br><code>abnormal = rhythm_label != "N"</code></div>
              <div class="kv-key">Métricas clave</div>
              <div class="kv-val">F1-macro (penaliza fallo en ambas clases) y Balanced Accuracy</div>
            </div>""",
            unsafe_allow_html=True,
        )

with col_e:
    with st.container(border=True):
        card_header("Pipeline resumido", "end-to-end")
        steps = [
            ("01", "Carga y EDA",           "VitalDB Arrhythmia · anotaciones PhysioNet"),
            ("02", "Limpieza y filtros",     "Noise excluido · calidad de señal"),
            ("03", "Features tabulares",     "Metadatos clínicos + RR intervals por latido"),
            ("04", "Feature selection",      f"{n_features_orig} features ({n_num_feat} num + {n_cat_feat} cat)"),
            ("05", "Split por case_id",      "80/20 · GroupShuffleSplit · sin leakage"),
            ("06", "Benchmark exploratorio", f"{n_models} candidatos · 150 casos · selección de modelo"),
            ("07", "Corrida final",          "Linear SVC · dataset completo · 639 460 registros"),
            ("08", "App Streamlit",          "Predicción demo sobre datos tabulares"),
        ]
        rows_html = "".join(
            f"""<div style="display:grid;grid-template-columns:28px 1fr;gap:8px 10px;
                           align-items:start;padding:6px 0;border-bottom:1px dashed var(--line-1)">
                  <div style="font-family:var(--mono);font-size:10px;color:var(--fg-4);
                              background:var(--bg-3);border-radius:4px;padding:2px 4px;
                              text-align:center;margin-top:2px">{n}</div>
                  <div>
                    <div style="font-size:12.5px;font-weight:500;color:var(--fg-0)">{name}</div>
                    <div style="font-size:11px;color:var(--fg-3);font-family:var(--mono)">{sub}</div>
                  </div>
                </div>"""
            for n, name, sub in steps
        )
        st.markdown(rows_html, unsafe_allow_html=True)

# ── NOTA TÉCNICA ───────────────────────────────────────────────────────────────
if meta and "library_versions" in meta:
    section_title("Entorno técnico")
    libs = meta["library_versions"]
    chips = " ".join(badge(f"{k} {v}", "muted") for k, v in libs.items())
    st.markdown(f'<div style="line-height:2.4">{chips}</div>', unsafe_allow_html=True)

page_footer()
