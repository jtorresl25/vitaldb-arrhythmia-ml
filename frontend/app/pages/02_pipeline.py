"""Página 02 — Pipeline metodológico · detección binaria normal/anormal."""

import streamlit as st

from components.badges import badge, badge_row
from components.cards  import callout, card_header, kv_table, section_title
from components.layout import page_header, page_footer
from utils.loaders     import load_model_metadata

# ── Metadata (live cuando disponible, fallback a valores del pipeline) ─────────
meta        = load_model_metadata()
winner_raw  = meta.get("winner_model", "linear_svc")           if meta else "linear_svc"
winner_nice = winner_raw.replace("_", " ").title()
f1_val      = meta.get("winner_test_f1_macro", 0.615)          if meta else 0.615
n_num       = len(meta.get("numeric_features",   []))          if meta else 57
n_cat       = len(meta.get("categorical_features", []))        if meta else 16
n_feat_orig = n_num + n_cat or 73
f1_str      = f"{f1_val:.3f}"

# ── Header ─────────────────────────────────────────────────────────────────────
page_header(
    "Pipeline metodológico — Detección binaria normal/anormal",
    "Flujo reproducible desde anotaciones VitalDB y metadatos clínicos hasta un "
    "modelo binario desplegable en Streamlit.",
    badge_html=badge("Metodología", "info"),
)

st.markdown(
    badge_row(
        badge("Binario", "ok"),
        badge("Sin leakage", "info"),
        badge("Group split", "info"),
        badge("Linear SVC", "muted"),
        badge("Streamlit ready", "teal"),
        badge("Demo académica", "warn"),
    ),
    unsafe_allow_html=True,
)

st.write("")

# ── Antes vs Ahora ─────────────────────────────────────────────────────────────
section_title("Cambio metodológico importante")

col_before, col_arrow, col_after = st.columns([5, 1, 5])

with col_before:
    with st.container(border=True):
        st.markdown(
            f'<div style="margin-bottom:6px">{badge("Antes", "warn")}</div>'
            f'<div style="font-size:13px;font-weight:600;color:var(--fg-0);">'
            f'Clasificación multiclase de ritmos</div>',
            unsafe_allow_html=True,
        )
        kv_table([
            ("Clases", "N, AFIB/AFL, VT, SVTA, AVB…"),
            ("Salida", "Tipo específico de arritmia"),
            ("Problema", "Desbalance severo · clases raras sin detección"),
            ("F1-macro", "Muy bajo en clases minoritarias"),
        ])

with col_arrow:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:center;'
        'height:100%;padding-top:36px;font-size:22px;color:var(--fg-3)">→</div>',
        unsafe_allow_html=True,
    )

with col_after:
    with st.container(border=True):
        st.markdown(
            f'<div style="margin-bottom:6px">{badge("Ahora", "ok")}</div>'
            f'<div style="font-size:13px;font-weight:600;color:var(--fg-0);">'
            f'Detección binaria normal/anormal</div>',
            unsafe_allow_html=True,
        )
        kv_table([
            ("Clases", "Normal · Anormal (2 clases)"),
            ("Salida", "¿El registro es normal o anormal?"),
            ("Ventaja", "Evaluación estable · métricas interpretables"),
            ("F1-macro", f1_str),
        ])

callout(
    "info",
    "Por qué se reformuló",
    "El enfoque multiclase tenía desempeño limitado y difícil de defender: clases como "
    "AVB (F1 = 0.000) o SVTA (F1 = 0.027) no eran detectadas de forma confiable. "
    "La reformulación binaria es más coherente con una <b>demo académica reproducible</b> "
    "y permite una evaluación más estable con las features tabulares disponibles.",
)

st.write("")

# ── Pipeline — pasos ──────────────────────────────────────────────────────────
section_title("Pasos del pipeline")

_STEPS = [
    (
        "01", "Datos de entrada", "teal",
        "VitalDB · anotaciones por latido · metadata clínica",
        [
            ("Señal",      "ECG Lead II · 500 Hz · archivos .npy por caso"),
            ("Anotaciones","Annotation_file_XXXX.csv · rhythm_label por latido"),
            ("Metadata",   "~25 variables clínicas/quirúrgicas por caso"),
            ("ECG crudo",  "Solo para visualización en la demo — no entra al modelo"),
        ],
    ),
    (
        "02", "Limpieza y filtros", "warn",
        "bad_signal · Noise · NaN · leakage excluidos",
        [
            ("Excluir",   "bad_signal_quality == True"),
            ("Excluir",   "rhythm_label == 'Noise'"),
            ("Excluir",   "rhythm_label vacío / NaN"),
            ("Excluir",   "beat_type — data leakage"),
            ("Conservar", "rhythm_label solo para construir target"),
        ],
    ),
    (
        "03", "Construcción del target binario", "ok",
        "normal (N) · anormal (≠ N)",
        [
            ("normal",   "rhythm_label == 'N'  →  392 623 registros"),
            ("anormal",  "rhythm_label != 'N'  →  246 837 registros"),
            ("Nota",     "Las clases originales se agrupan como 'anormal'"),
            ("Objetivo", "Detectar si el registro es normal o anormal"),
        ],
    ),
    (
        "04", "Feature engineering tabular", "info",
        f"RR intervals · metadata clínica · {n_feat_orig} vars → 162 tras OHE",
        [
            ("RR",        "rr_prev · rr_next · hr_inst_from_rr_prev"),
            ("Posición",  "position_in_case (0 = inicio, 1 = fin)"),
            ("Clínicas",  f"{n_num} numéricas + {n_cat} categóricas"),
            ("OHE",       "162 variables transformadas"),
        ],
    ),
    (
        "05", "Split sin leakage", "info",
        "GroupShuffleSplit · 80/20 por case_id · 482 casos",
        [
            ("Estrategia", "GroupShuffleSplit por case_id"),
            ("Train",      "510 287 registros · 385 casos"),
            ("Test",       "129 173 registros · 97 casos"),
            ("Garantía",   "Ningún caso aparece en train y test a la vez"),
        ],
    ),
    (
        "06", "Benchmark exploratorio", "muted",
        "5 candidatos · 150 casos · orientación de selección",
        [
            ("Modelos",   "LogReg · LinearSVC · DTree · RandomForest · MLP"),
            ("Config",    "--max-cases 150 · --n-iter 5 · --n-splits 3"),
            ("Propósito", "Comparar candidatos · no métricas finales"),
            ("Resultado", "MLP mayor F1 exploratorio · no elegido como final"),
        ],
    ),
    (
        "07", "Modelo final oficial", "teal",
        f"Linear SVC · dataset completo · F1-macro {f1_str}",
        [
            ("Modelo",    winner_nice),
            ("Dataset",   "639 460 registros · 482 casos"),
            ("F1-macro",  f1_str),
            ("Bal. acc.", "0.616"),
            ("Criterio",  "Desempeño · interpretabilidad · velocidad"),
        ],
    ),
    (
        "08", "Exportación y despliegue", "muted",
        ".joblib · metadata JSON · artefactos → Streamlit Cloud",
        [
            ("Pipeline",    "tabular_best_model_pipeline.joblib"),
            ("Metadata",    "tabular_best_model_metadata.json"),
            ("Reportes",    "CSV con métricas · classification report"),
            ("Artefactos",  "frontend/app/app_artifacts/ para Cloud"),
        ],
    ),
]

for step_num, step_name, accent, step_sub, step_kv in _STEPS:
    with st.container(border=True):
        col_n, col_c = st.columns([0.45, 9.55])
        with col_n:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:12px;font-weight:700;'
                f'color:var(--fg-4);background:var(--bg-3);border-radius:5px;'
                f'padding:4px 6px;text-align:center;margin-top:6px">{step_num}</div>',
                unsafe_allow_html=True,
            )
        with col_c:
            col_title, col_detail = st.columns([1.6, 2])
            with col_title:
                st.markdown(
                    f'<div style="font-size:13.5px;font-weight:600;color:var(--fg-0);'
                    f'margin-bottom:3px">{step_name}</div>'
                    f'<div style="font-size:10.5px;font-family:var(--mono);'
                    f'color:var(--fg-3)">{step_sub}</div>',
                    unsafe_allow_html=True,
                )
            with col_detail:
                kv_table(step_kv)

st.write("")

# ── Qué entra y qué no entra ──────────────────────────────────────────────────
section_title("Qué entra y qué no entra al modelo")

col_in, col_out = st.columns(2)

with col_in:
    with st.container(border=True):
        st.markdown(
            f'<div style="margin-bottom:8px">{badge("✓  Entra al modelo", "ok")}</div>',
            unsafe_allow_html=True,
        )
        kv_table([
            ("RR intervals",       "rr_prev · rr_next · hr_inst_from_rr_prev"),
            ("Temporal",           "time_second · position_in_case"),
            ("Metadata clínica",   "bmi · preop_* · intraop_crystalloid…"),
            ("Metadata quirúrgica","optype · opstart · opend · anestart…"),
            ("Categóricas (OHE)",  "optype · iv1 · aline1 · cline1"),
        ])

with col_out:
    with st.container(border=True):
        st.markdown(
            f'<div style="margin-bottom:8px">{badge("✗  No entra al modelo", "err")}</div>',
            unsafe_allow_html=True,
        )
        kv_table([
            ("rhythm_label",    "Es el target — leakage directo"),
            ("beat_type",       "Descriptor del latido — leakage potencial"),
            ("case_id",         "Solo para split; no es predictor"),
            ("subjectid / dx",  "Alta cardinalidad / no disponible en demo"),
            ("death_inhosp",    "Resultado tardío — no disponible al predecir"),
            ("ECG crudo",       "No alimenta el modelo final directamente"),
        ])

st.write("")

# ── Decisiones de diseño ──────────────────────────────────────────────────────
section_title("Decisiones de diseño clave")

col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        card_header("¿Por qué features tabulares?", "no CNN ni transformers")
        kv_table([
            ("Ventaja",    "Interpretable · rápido · reproducible con scikit-learn"),
            ("Limitación", "No captura morfología de la onda ECG"),
            ("Alternativa","1D-CNN sobre fragmentos de señal — trabajo futuro"),
        ])
        callout(
            "info",
            "",
            "El enfoque extrae <b>RR intervals + metadatos clínicos</b> en un vector "
            "interpretable, evitando el alto costo computacional de modelos sobre señal cruda.",
        )

with col_b:
    with st.container(border=True):
        card_header("¿Por qué F1-macro?", "dataset desbalanceado: ~61 % normal")
        kv_table([
            ("Accuracy",    "Engañosa: un modelo que siempre predice Normal alcanza ~61 %"),
            ("F1-macro",    f"Penaliza el fallo en ambas clases — valor: {f1_str}"),
            ("Bal. acc.",   "0.616 · independiente del desbalance"),
        ])
        callout(
            "warn",
            "",
            "Si el modelo siempre prediciese <b>Normal</b>, accuracy sería ~61 % "
            "pero <b>nunca detectaría un caso anormal</b>. F1-macro y balanced accuracy "
            "evitan que eso se oculte.",
        )

st.write("")

# ── Artefactos generados ──────────────────────────────────────────────────────
section_title("Artefactos generados por el pipeline")

_ARTIFACTS = [
    ("models/",
     "tabular_best_model_pipeline.joblib",
     "Pipeline sklearn final (preprocesador + LinearSVC)",
     "teal"),
    ("models/",
     "tabular_best_model_metadata.json",
     "Metadata: features, métricas, hiperparámetros",
     "teal"),
    ("reports/tables/",
     "tabular_model_comparison_test.csv",
     "Métricas de la corrida final",
     "info"),
    ("reports/tables/",
     "tabular_model_comparison_history.csv",
     "Benchmark exploratorio de candidatos",
     "info"),
    ("reports/tables/",
     "tabular_best_model_classification_report.csv",
     "Reporte normal / anormal",
     "info"),
    ("reports/tables/",
     "tabular_binary_metrics.csv",
     "Métricas binarias específicas",
     "info"),
    ("reports/tables/",
     "confusion_matrix.csv",
     "Matriz normal / anormal (long format)",
     "muted"),
    ("frontend/app/",
     "app_artifacts/",
     "Carpeta versionable para Streamlit Cloud — artefactos pequeños copiados del pipeline",
     "ok"),
]

_art_cols = st.columns(2)
for i, (folder, fname, desc, accent) in enumerate(_ARTIFACTS):
    with _art_cols[i % 2]:
        st.markdown(
            f'<div style="display:flex;gap:8px;align-items:flex-start;padding:7px 0;'
            f'border-bottom:1px dashed var(--line-1)">'
            f'<div style="margin-top:2px">{badge("↓", accent)}</div>'
            f'<div style="min-width:0">'
            f'<div style="font-family:var(--mono);font-size:10px;color:var(--fg-3)">'
            f'{folder}<b style="color:var(--fg-1)">{fname}</b></div>'
            f'<div style="font-size:11px;color:var(--fg-2);margin-top:2px">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# ── Métricas finales del modelo ───────────────────────────────────────────────
section_title("Métricas finales — modelo oficial")

_met_cols = st.columns(5)
_METRICS = [
    ("F1-macro",      f1_str,  "teal"),
    ("Balanced acc.", "0.616", "teal"),
    ("Accuracy",      "0.633", "blue"),
    ("Precision",     "0.615", "blue"),
    ("Recall",        "0.616", "blue"),
]
for _col, (name, val, accent) in zip(_met_cols, _METRICS):
    with _col:
        st.markdown(
            f'<div style="border:1px solid var(--line-1);border-radius:8px;'
            f'padding:10px 12px;text-align:center;">'
            f'<div style="font-family:var(--mono);font-size:11px;color:var(--fg-3);'
            f'margin-bottom:4px">{name}</div>'
            f'<div style="font-size:22px;font-weight:700;color:var(--{accent})">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.write("")

# ── Nota de transparencia ─────────────────────────────────────────────────────
callout(
    "warn",
    "Transparencia metodológica",
    "<b>ECG crudo:</b> visible en la demo para contexto visual, pero no alimenta directamente "
    "al modelo final — el modelo usa features tabulares (RR + metadata clínica). "
    "<b>Importancia de features:</b> refleja asociaciones predictivas aprendidas, no causalidad médica. "
    "<b>Uso clínico:</b> esta app es académica — no debe usarse para diagnóstico ni decisiones clínicas.",
)

page_footer()
