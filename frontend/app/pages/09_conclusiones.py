"""Página 09 — Conclusiones · cierre del proyecto binario Normal/Anormal."""

import streamlit as st
import numpy as np
import pandas as pd

from components.layout import page_header, page_footer
from components.cards  import callout, card_header, kv_table, metric_card, section_title
from components.badges import badge, badge_row
from utils.loaders import (
    load_model_metadata,
    load_model_comparison_history,
    load_feature_importance,
    load_binary_metrics,
    load_train_test_split_summary,
)

# ── Model display names ────────────────────────────────────────────────────────
_MODEL_DISPLAY = {
    "linear_svc":          "Linear SVC",
    "random_forest":       "Random Forest",
    "mlp":                 "MLP",
    "decision_tree":       "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "logreg":              "Logistic Regression",
    "xgboost":             "XGBoost",
}

# ── Bootstrap ──────────────────────────────────────────────────────────────────
meta       = load_model_metadata()
df_hist    = load_model_comparison_history()
df_imp     = load_feature_importance()
df_bm      = load_binary_metrics()
split_info = load_train_test_split_summary()

_winner_id   = (meta.get("winner_model", "linear_svc") if meta else "linear_svc") or "linear_svc"
_winner_nice = _MODEL_DISPLAY.get(_winner_id, _winner_id.replace("_", " ").title())
f1_val       = (meta.get("winner_test_f1_macro", 0.615) if meta else 0.615) or 0.615
f1_str       = f"{f1_val:.3f}"

n_num   = len(meta.get("numeric_features",   [])) if meta else 57
n_cat   = len(meta.get("categorical_features", [])) if meta else 16
n_orig  = n_num + n_cat or 73


def _int(d, key, fb):
    v = (d or {}).get(key)
    try:    return int(float(v))
    except: return fb


n_train_grp = _int(split_info, "n_train_groups", 385)
n_test_grp  = _int(split_info, "n_test_groups",  97)
train_rows  = _int(split_info, "n_train_rows",   510287)
test_rows   = _int(split_info, "n_test_rows",    129173)
total_rows  = train_rows + test_rows


def _bm(key, fb=None):
    if df_bm is None or df_bm.empty or key not in df_bm.columns:
        return fb
    v = df_bm.iloc[0][key]
    try:
        f = float(v)
        return fb if np.isnan(f) else f
    except (TypeError, ValueError):
        return fb


acc     = _bm("accuracy",                    0.633)
sens    = _bm("sensitivity_recall_abnormal", 0.539)
spec    = _bm("specificity_recall_normal",   0.692)
prec    = _bm("precision_abnormal",          0.527)
f1_abn  = _bm("f1_abnormal",                 0.533)
bal_acc = round((sens + spec) / 2, 3) if (sens and spec) else 0.616
tn      = int(_bm("tn_normal",               54625))
tp      = int(_bm("tp_abnormal",             27082))
fn      = int(_bm("fn_abnormal_missed",      23180))
fp      = int(_bm("fp_abnormal_false_alarm",  24286))

n_models = len(df_hist) if df_hist is not None else 5

# Top features for summary text
_fc = _ic = None
_top_feats_html = "features tabulares (RR intervals + metadatos clínicos)"
if df_imp is not None and not df_imp.empty:
    _fc = next((c for c in ["feature_base", "feature"] if c in df_imp.columns), None)
    _ic = next((c for c in ["importance", "abs_coef"]  if c in df_imp.columns), None)
    if _fc and _ic:
        _top2 = (df_imp.dropna(subset=[_ic])
                       .sort_values(_ic, ascending=False)
                       .head(2)[_fc].tolist())
        if _top2:
            _clean = [
                str(n).replace("cat__", "").replace("num__", "").replace("_", " ")[:28]
                for n in _top2
            ]
            _top_feats_html = (
                f"<code>{_clean[0]}</code>"
                + (f" · <code>{_clean[1]}</code>" if len(_clean) > 1 else "")
            )

# HP string
def _fmt_param_value(v) -> str:
    try:
        if isinstance(v, bool):   return str(v)
        if isinstance(v, float):  return f"{v:.4g}"
        if isinstance(v, int):    return str(v)
    except Exception:             pass
    return str(v)

_hp_str = "—"
if meta and "best_hyperparams_per_model" in meta:
    _hp = meta["best_hyperparams_per_model"].get(_winner_id, {})
    if _hp:
        _hp_str = " · ".join(
            f"{k}={_fmt_param_value(v)}" for k, v in list(_hp.items())[:2]
        )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
page_header(
    "Conclusiones",
    "Resumen final del modelo binario Normal/Anormal, sus alcances, "
    "limitaciones y próximos pasos.",
    badge_html=badge_row(badge("Cierre del proyecto", "info"), badge("Académico", "muted")),
)

callout(
    "warn",
    "Alcance académico — no clínico",
    "Esta app es una <b>demo académica</b> de Machine Learning. "
    "El modelo detecta registros <b>Normal</b> o <b>Anormal</b> — "
    "<b>no diagnostica el tipo específico de arritmia</b>. "
    "No reemplaza la interpretación médica ni debe usarse para decisiones clínicas.",
)

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
section_title("Resumen ejecutivo")

kv_table([
    ("Tarea",
     "Clasificación binaria: <b>Normal</b> (rhythm_label = N) "
     "vs <b>Anormal</b> (rhythm_label ≠ N)"),
    ("Modelo final oficial",
     f"<b>{_winner_nice}</b> — seleccionado por desempeño, estabilidad, "
     "interpretabilidad y compatibilidad con la app"),
    ("Dataset",
     f"{total_rows:,} registros · 482 casos · "
     f"{n_orig} features originales ({n_num} num + {n_cat} cat) · 162 tras OHE"),
    ("Split",
     f"80/20 por case_id · {n_train_grp} grupos train ({train_rows:,}) "
     f"/ {n_test_grp} test ({test_rows:,}) · sin leakage entre pacientes"),
    ("Métricas principales",
     f"F1-macro = {f1_str} · Balanced acc. = {bal_acc:.3f} · "
     f"Sensibilidad (Anormal) = {sens:.3f} · Especificidad (Normal) = {spec:.3f}"),
    ("Benchmark",
     f"Se compararon {n_models} candidatos en corrida exploratoria (150 casos). "
     "Modelo final entrenado con el dataset completo."),
    ("Interpretabilidad",
     f"Coeficientes {_winner_nice} disponibles · "
     f"features más relevantes: {_top_feats_html}"),
])

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTADOS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
section_title("Resultados principales")

c1, c2, c3, c4 = st.columns(4)

with c1:
    with st.container(border=True):
        card_header("Modelo final oficial", _winner_nice,
                    right_html=badge("oficial", "teal"))
        metric_card("F1-macro (test)", f1_str,
                    "métrica principal", accent="teal", helper_kind="ok")
        metric_card("Balanced accuracy", f"{bal_acc:.3f}",
                    "robusta ante desbalance", accent="teal")
        metric_card("Hiperparámetros",
                    _hp_str[:36] + ("…" if len(_hp_str) > 36 else ""),
                    "selección final", accent="muted")

with c2:
    with st.container(border=True):
        card_header("Dataset tabular", f"{total_rows:,} registros",
                    right_html=badge("Tabular", "info"))
        metric_card("Train", f"{train_rows:,}",
                    f"{n_train_grp} grupos (casos)", accent="blue")
        metric_card("Test",  f"{test_rows:,}",
                    f"{n_test_grp} grupos (casos)", accent="blue")
        metric_card("Features", f"{n_orig} orig · 162 OHE",
                    f"{n_num} numéricas + {n_cat} categóricas", accent="muted")

with c3:
    with st.container(border=True):
        card_header("Métricas binarias", "clase positiva = Anormal",
                    right_html=badge("binario", "ok"))
        metric_card("Sensibilidad",  f"{sens:.3f}" if sens else "—",
                    "recall Anormal", accent="warn")
        metric_card("Especificidad", f"{spec:.3f}" if spec else "—",
                    "recall Normal", accent="info")
        metric_card("F1 Anormal",    f"{f1_abn:.3f}" if f1_abn else "—",
                    f"TP={tp:,} · FN={fn:,}", accent="err")

with c4:
    with st.container(border=True):
        card_header("Interpretabilidad", f"coeficientes {_winner_nice}",
                    right_html=badge("global", "muted"))
        if df_imp is not None and not df_imp.empty and _fc and _ic:
            _top3 = (df_imp.dropna(subset=[_ic])
                           .sort_values(_ic, ascending=False)
                           .head(3))
            for _rank in range(min(3, len(_top3))):
                _fname = (str(_top3.iloc[_rank][_fc])
                          .replace("cat__", "").replace("num__", "")
                          .replace("_", " ")[:22])
                _fval  = float(_top3.iloc[_rank][_ic])
                metric_card(
                    f"Feature #{_rank + 1}", _fname,
                    f"abs_coef {_fval:.3f}",
                    accent="teal" if _rank == 0 else "blue",
                    helper_kind="ok" if _rank == 0 else "muted",
                )
        else:
            metric_card("Features", "—", "importancia no disponible", accent="muted")

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# POR QUÉ NO SE ELIGIÓ MLP
# ══════════════════════════════════════════════════════════════════════════════
section_title("Por qué no se eligió MLP como modelo final")

with st.container(border=True):
    card_header(
        "MLP — buen desempeño exploratorio, no seleccionado",
        "corrida exploratoria · 150 casos",
        right_html=badge("descartado", "warn"),
    )
    kv_table([
        ("Desempeño exploratorio",
         "MLP obtuvo el mayor F1-macro en la corrida con 150 casos y parámetros reducidos."),
        ("Problema 1 — datos insuficientes",
         "La corrida exploratoria <b>no usó el dataset completo</b> — "
         "las métricas no son representativas del rendimiento real."),
        ("Problema 2 — convergencia",
         "Presentó <b>advertencias de convergencia</b> "
         "(<code>ConvergenceWarning</code>) en múltiples folds."),
        ("Problema 3 — interpretabilidad",
         "No permite extraer importancia de variables de forma directa — "
         "<b>sin interpretabilidad nativa</b>."),
        ("Decisión final",
         f"<b>{_winner_nice}</b> ofrece mejor equilibrio entre desempeño, "
         "estabilidad, interpretabilidad y velocidad. "
         "Para una demo académica defendible, es la elección más sólida."),
    ])

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# LO QUE FUNCIONÓ BIEN
# ══════════════════════════════════════════════════════════════════════════════
section_title("Lo que funcionó bien")

with st.container(border=True):
    card_header("Aciertos del pipeline", "evaluación honesta")
    kv_table([
        (f'{badge("✓", "ok")} Pipeline reproducible',
         "Cada paso está en notebooks numerados y es re-ejecutable desde cero."),
        (f'{badge("✓", "ok")} Split por case_id',
         "La separación por caso evita leakage entre registros del mismo paciente."),
        (f'{badge("✓", "ok")} Reformulación binaria',
         "Pasar de multiclase a Normal/Anormal produjo un modelo "
         "más estable e interpretable."),
        (f'{badge("✓", "ok")} Benchmark exploratorio',
         f"Se compararon {n_models} candidatos con búsqueda de "
         "hiperparámetros antes de la corrida final."),
        (f'{badge("✓", "ok")} Interpretabilidad',
         f"Los coeficientes de {_winner_nice} permiten identificar "
         "qué variables clínicas son más discriminativas."),
        (f'{badge("✓", "ok")} Evaluación por caso',
         "Se calcularon métricas por case_id para "
         "seleccionar casos demo representativos."),
        (f'{badge("✓", "ok")} Artefactos para despliegue',
         "Carpeta <code>app_artifacts/</code> con archivos livianos "
         "lista para Streamlit Cloud."),
    ])

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# LIMITACIONES
# ══════════════════════════════════════════════════════════════════════════════
section_title("Principales limitaciones")

lim1, lim2 = st.columns(2)

_fn_rate = f"{fn / (fn + tp):.0%}" if (fn + tp) > 0 else "—"

with lim1:
    callout(
        "warn",
        f"Desempeño moderado — F1-macro = {f1_str}",
        "El modelo no alcanza calidad clínica. "
        f"La sensibilidad de Anormal ({sens:.1%}) implica que el {_fn_rate} "
        "de los registros anormales no son detectados (FN).",
    )
    callout(
        "warn",
        "Predicción binaria, no diagnóstico de arritmia",
        "El modelo detecta Normal vs Anormal. "
        "<b>No identifica el tipo de arritmia</b> (AFIB, VT, AVB, etc.). "
        "Las etiquetas originales solo se usaron para construir la clase Anormal.",
    )
    callout(
        "warn",
        "Features tabulares, no señal ECG cruda",
        "El modelo usa RR intervals y metadatos clínicos. "
        "La morfología de la onda ECG no se captura directamente.",
    )

with lim2:
    callout(
        "warn",
        "Asociaciones predictivas, no causalidad",
        "La importancia de features refleja correlaciones aprendidas del dataset, "
        "no mecanismos causales clínicos.",
    )
    callout(
        "warn",
        "Sin validación externa",
        "Los resultados se obtuvieron con split interno por case_id. "
        "Falta validación en datos de otras instituciones o equipos.",
    )
    callout(
        "warn",
        "Variables acumuladas intraoperatorias",
        "Algunas features (p. ej. <code>intraop_crystalloid</code>) "
        "representan totales del caso, no el estado en el momento del latido — "
        "posible sesgo temporal.",
    )

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# PRÓXIMOS PASOS
# ══════════════════════════════════════════════════════════════════════════════
section_title("Próximos pasos — hoja de ruta")

col_near, col_far = st.columns(2)

with col_near:
    with st.container(border=True):
        card_header("Corto plazo", "completar el despliegue",
                    right_html=badge("prioridad alta", "err"))
        kv_table([
            ("1", "Verificar que <code>app_artifacts/</code> esté completo "
                  "y Streamlit Cloud cargue modelos y CSV correctamente."),
            ("2", "Mejorar la visualización de ECG: tramos coloreados en rojo "
                  "para predicciones Anormales."),
            ("3", "Validar disponibilidad del archivo <code>case_337.npy</code> "
                  "para el caso demo mixto."),
            ("4", "Revisar página 03 (Dataset y limpieza) para coherencia "
                  "con el pipeline binario final."),
        ])

with col_far:
    with st.container(border=True):
        card_header("Mediano plazo", "mejorar el modelo",
                    right_html=badge("calidad", "info"))
        kv_table([
            ("5", "<b>Ajuste de umbral</b>: explorar umbrales &lt; 0.5 "
                  "para aumentar recall de Anormal."),
            ("6", "<b>Calibración</b>: agregar "
                  "<code>CalibratedClassifierCV</code> para probabilidades calibradas."),
            ("7", "<b>Features clínicas</b>: evaluar selección de variables "
                  "con criterios clínicos, no solo estadísticos."),
            ("8", "<b>SHAP</b>: reemplazar importancia global por "
                  "explicaciones locales por predicción."),
            ("9", "<b>Validación externa</b>: evaluar en datos de "
                  "otras instituciones."),
            ("10", "<b>1D-CNN</b>: explorar modelos directamente sobre señal ECG "
                   "como línea de investigación futura."),
        ])

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# REFLEXIÓN FINAL
# ══════════════════════════════════════════════════════════════════════════════
section_title("Reflexión final")

with st.container(border=True):
    card_header(
        "Balance del proyecto",
        "lo que logramos y lo que falta",
        right_html=badge("académico", "muted"),
    )
    st.write("")
    st.html(
        f'<p style="color:var(--fg-1);font-size:15px;line-height:1.75;max-width:860px">'
        f"El proyecto demuestra un <b>flujo académico reproducible</b> para detectar "
        f"registros normales y anormales en anotaciones intraoperatorias de ECG "
        f"usando features tabulares derivadas de RR intervals y metadatos clínicos. "
        f"Aunque el desempeño no es suficiente para uso clínico, la "
        f"<b>reformulación binaria</b> permitió construir una demo más estable e "
        f"interpretable que el enfoque multiclase inicial. "
        f"El modelo final, <b>{_winner_nice}</b>, ofrece un balance razonable entre "
        f"desempeño, simplicidad e interpretabilidad, y queda desplegable en Streamlit "
        f"mediante artefactos livianos. "
        f"Los próximos pasos apuntan a ajuste de umbral, calibración de probabilidades "
        f"y exploración de validación externa."
        f"</p>"
    )
    st.write("")
    callout(
        "info",
        "Reproducibilidad",
        "Todo el código (pipeline, modelos y app) puede re-ejecutarse desde cero "
        "siguiendo los notebooks numerados del repositorio.",
    )

page_footer()
