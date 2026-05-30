"""Página 03 — Dataset y limpieza."""

import streamlit as st

from components.badges import badge, badge_row
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.layout import page_header, page_footer
from utils.loaders import load_classification_report, load_model_metadata

page_header(
    "Dataset y limpieza",
    "Auditoría de calidad, filtros aplicados y construcción del dataset tabular.",
    badge_html=badge("Metodología de datos", "info"),
)

st.markdown(
    badge_row(
        badge("VitalDB Arrhythmia DB", "info"),
        badge("PhysioNet annotations", "info"),
        badge("Tabular features", "muted"),
        badge("GroupSplit sin leakage", "warn"),
    ),
    unsafe_allow_html=True,
)

st.write("")

meta   = load_model_metadata()
df_cls = load_classification_report()

# ── 1 · Fuente de datos ───────────────────────────────────────────────────────
section_title("Fuente de datos")

col_src, col_ann = st.columns(2)

with col_src:
    with st.container(border=True):
        card_header("VitalDB Arrhythmia Database", "señales ECG intraoperatorias")
        kv_table([
            ("Tipo de señal",    "ECG Lead II · 500 Hz de muestreo"),
            ("Escenario",        "Pacientes quirúrgicos en sala de operaciones"),
            ("Formato",          "Archivos .npy por caso (array NumPy 1D)"),
            ("Anotaciones",      "PhysioNet · ritmo por latido latido (CSV)"),
            ("Cobertura",        "Miles de casos con diversas duraciones y ritmos"),
            ("Variables clínicas","metadata.csv con ~25 campos por caso"),
        ])

with col_ann:
    with st.container(border=True):
        card_header("Archivos de anotación", "PhysioNet · Annotation_file_XXXX.csv")
        st.markdown(
            "Cada caso tiene un archivo CSV con una fila por latido detectado. "
            "Las columnas principales son:",
            unsafe_allow_html=False,
        )
        kv_table([
            ("time_second",       "Instante del latido en segundos desde el inicio"),
            ("rhythm_label",      "Clase de ritmo en ese latido (N, AFIB/AFL, VT…)"),
            ("beat_type",         "Tipo de latido (N, V, A, S…) — solo descriptivo"),
            ("bad_signal_quality","Indicador booleano de artefacto / mala calidad"),
        ])
        callout(
            "warn",
            "beat_type excluido del modelo",
            "<code>beat_type</code> es un descriptor del latido individual, "
            "no un predictor de ritmo válido. Incluirlo causaría fuga de información "
            "(data leakage) porque está derivado del mismo proceso de anotación.",
        )

st.write("")

# ── 2 · Filtros de limpieza ───────────────────────────────────────────────────
section_title("Filtros de calidad aplicados")

_FILTROS = [
    ("F1", "Señal de mala calidad",
     "bad_signal_quality == True",
     "Latidos marcados como artefacto o segmento no interpretable son excluidos del dataset.",
     "err"),
    ("F2", "Clase 'Noise'",
     "rhythm_label == 'Noise'",
     "La etiqueta Noise no representa un ritmo cardíaco real. Se excluye por metodología del proyecto.",
     "err"),
    ("F3", "rhythm_label vacío o NaN",
     "rhythm_label.isna() o valor vacío / 'none'",
     "Filas sin etiqueta de ritmo son eliminadas. No se imputan etiquetas.",
     "err"),
    ("F4", "Clases con soporte mínimo insuficiente",
     "clases con muy pocos latidos en el dataset completo",
     "Clases con soporte extremadamente bajo pueden eliminarse según configuración del pipeline.",
     "warn"),
]

for code, name, condition, desc, accent in _FILTROS:
    with st.container(border=True):
        col_code, col_body = st.columns([0.4, 9.6])
        with col_code:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:11px;font-weight:700;'
                f'background:var(--bg-3);border-radius:4px;padding:3px 6px;'
                f'text-align:center;margin-top:4px;color:var(--fg-3)">{code}</div>',
                unsafe_allow_html=True,
            )
        with col_body:
            st.markdown(
                f'<div style="font-size:13px;font-weight:600;color:var(--fg-0)'
                f';margin-bottom:2px">{name}</div>'
                f'<div style="font-family:var(--mono);font-size:11px;color:var(--fg-3)'
                f';margin-bottom:4px">Condición: {condition}</div>'
                f'<div style="font-size:12px;color:var(--fg-2)">{desc}</div>',
                unsafe_allow_html=True,
            )

st.write("")

# ── 3 · Feature engineering ───────────────────────────────────────────────────
section_title("Construcción de features tabulares")

col_feat_a, col_feat_b = st.columns(2)

with col_feat_a:
    with st.container(border=True):
        card_header("Features de anotaciones (5)", "calculadas por latido")
        kv_table([
            ("time_second",           "Tiempo absoluto del latido en el caso"),
            ("rr_prev",               "Intervalo RR con el latido anterior (s)"),
            ("rr_next",               "Intervalo RR con el latido siguiente (s)"),
            ("hr_inst_from_rr_prev",  "Frecuencia cardíaca instantánea = 60 / rr_prev"),
            ("position_in_case",      "Posición relativa en el caso (0=inicio, 1=fin)"),
        ])

n_numeric = len(meta.get("numeric_features", [])) if meta else 13
n_categ   = len(meta.get("categorical_features", [])) if meta else 2

with col_feat_b:
    with st.container(border=True):
        card_header(
            f"Features de metadata clínica ({n_numeric + n_categ})",
            "broadcast de caso a todos sus latidos",
        )
        numeric_feats = meta.get("numeric_features", [
            "analyzed_duration_sec", "total_beats", "caseend", "anestart",
            "aneend", "opstart", "opend", "bmi", "preop_plt", "preop_pt",
            "preop_alb", "preop_cr", "intraop_crystalloid",
        ]) if meta else []
        categ_feats = meta.get("categorical_features", ["optype", "aline1"]) if meta else []

        if numeric_feats:
            st.markdown(
                f'<div style="font-size:11px;color:var(--fg-3);margin-bottom:4px">'
                f'<b>Numéricas ({len(numeric_feats)}):</b> '
                + ", ".join(f"<code>{f}</code>" for f in numeric_feats)
                + "</div>",
                unsafe_allow_html=True,
            )
        if categ_feats:
            st.markdown(
                f'<div style="font-size:11px;color:var(--fg-3)">'
                f'<b>Categóricas ({len(categ_feats)}):</b> '
                + ", ".join(f"<code>{f}</code>" for f in categ_feats)
                + "</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Los valores de metadata se repiten en todas las filas del mismo caso. "
            "Los valores faltantes se imputan con mediana/moda dentro del pipeline."
        )

st.write("")

# ── 4 · Split por case_id ─────────────────────────────────────────────────────
section_title("División train / test sin leakage")

with st.container(border=True):
    card_header("GroupShuffleSplit por case_id", "evita que el mismo paciente aparezca en train y test")
    col_split_a, col_split_b = st.columns(2)
    with col_split_a:
        kv_table([
            ("Estrategia",   "GroupShuffleSplit"),
            ("Grupo",        "case_id (un caso = un paciente quirúrgico)"),
            ("Proporción",   "80% train · 20% test"),
            ("Reproducible", "random_state fijo"),
        ])
    with col_split_b:
        callout(
            "info",
            "¿Por qué es crucial?",
            "Si latidos del mismo caso aparecen en train y test, el modelo aprende "
            "el 'estilo' de ese paciente y reporta accuracy inflada. "
            "El GroupShuffleSplit garantiza que cada caso aparezca <b>solo</b> en "
            "train <b>o</b> test, nunca en los dos.",
        )

st.write("")

# ── 5 · Clases (desde CSV si disponible) ──────────────────────────────────────
section_title("Distribución de clases")

if df_cls is not None:
    _summary = {"accuracy", "macro avg", "weighted avg"}
    df_c = df_cls[~df_cls.index.isin(_summary)].copy()
    import pandas as _pd
    for col in ["precision", "recall", "f1-score", "support"]:
        if col in df_c.columns:
            df_c[col] = _pd.to_numeric(df_c[col], errors="coerce")

    st.dataframe(
        df_c.style.format({
            "precision": "{:.3f}", "recall": "{:.3f}",
            "f1-score":  "{:.3f}", "support": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True,
    )
    st.caption("Reporte por clase desde el test set del modelo ganador.")
else:
    callout(
        "info",
        "Tabla de distribución no disponible en este entorno",
        "El archivo <code>reports/tables/tabular_best_model_classification_report.csv</code> "
        "se genera al ejecutar el pipeline de entrenamiento. "
        "Las clases presentes son: N, AFIB/AFL, Patterned Ventricular Ectopy, "
        "Patterned Atrial Ectopy, SVTA, VT, SND, AVB, WAP/MAT, Unclassifiable.",
    )

st.write("")

# ── 6 · Salidas del pipeline ───────────────────────────────────────────────────
section_title("Salidas generadas por el pipeline")

_OUTPUTS = [
    ("models/tabular_best_model_pipeline.joblib", "Pipeline sklearn serializado (preprocesador + clasificador)"),
    ("models/tabular_best_model_metadata.json",   "Metadata del modelo: features, métricas, hiperparámetros"),
    ("reports/tables/tabular_model_comparison_test.csv",           "Comparativa test de todos los modelos"),
    ("reports/tables/tabular_best_model_classification_report.csv","Reporte por clase en test set"),
    ("reports/tables/tabular_confusion_matrix_absolute.csv",       "Matriz de confusión en formato largo"),
    ("reports/tables/tabular_feature_importance_best_model.csv",   "Importancia de features del ganador"),
    ("reports/figures/tabular_best_model_confusion_matrix_absolute.png", "Imagen de la matriz de confusión"),
]

with st.container(border=True):
    for path, desc in _OUTPUTS:
        st.markdown(
            f'<div style="display:flex;gap:8px;padding:5px 0;'
            f'border-bottom:1px dashed var(--line-1);align-items:flex-start">'
            f'<code style="font-size:10px;color:var(--fg-3);min-width:0;flex:1;'
            f'word-break:break-all">{path}</code>'
            f'<div style="font-size:11px;color:var(--fg-2);min-width:200px;'
            f'max-width:240px;text-align:right">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

page_footer()
