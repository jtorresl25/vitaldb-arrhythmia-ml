"""Página 02 — Pipeline metodológico (Streamlit nativo)."""

import streamlit as st

from components.badges import badge, badge_row
from components.cards import callout, card_header, kv_table, section_title
from components.layout import page_header
from utils.loaders import load_model_metadata

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

st.write("")

meta       = load_model_metadata()
n_features = meta.get("n_features", 15) if meta else 15
winner     = meta.get("winner_model", "Random Forest").replace("_", " ").title() if meta else "Random Forest"
f1_str     = f"{meta.get('winner_test_f1_macro', 0):.3f}" if meta else "—"

# ── Flujo general ─────────────────────────────────────────────────────────────
section_title("Flujo completo del proyecto")

_STEPS = [
    ("01", "Problema y datos",
     "VitalDB Arrhythmia Database · anotaciones PhysioNet · ECG intraoperatorio",
     "Señales ECG continuas de pacientes quirúrgicos con anotaciones de ritmo latido a latido. "
     "La base contiene ~5 000 casos con distintos tipos de arritmia.",
     "info"),
    ("02", "Carga y exploración (EDA)",
     "metadata.csv · Annotation_files · señales .npy a 500 Hz",
     "Carga de metadatos clínicos (optype, altura, peso, laboratorios pre-op) y archivos de "
     "anotaciones de ritmo. Exploración de distribución de clases, duración de casos y calidad de señal.",
     "info"),
    ("03", "Filtros de calidad y limpieza",
     "bad_signal_quality · Noise excluido · ritmos válidos",
     "Se aplican filtros: exclusión de latidos con bad_signal_quality=True, exclusión de "
     "la clase 'Noise' y de rhythm_label vacío/NaN. Se conservan solo ritmos con soporte suficiente.",
     "warn"),
    ("04", "Feature engineering tabular",
     f"RR intervals · metadatos clínicos · {n_features} features totales",
     "Por cada latido se calculan: rr_prev, rr_next, hr_inst_from_rr_prev, position_in_case "
     "(features de anotaciones) + 25 campos de metadata clínica (optype, bmi, preop_*, intraop_*). "
     "beat_type se excluye del modelo — es descriptivo, no un predictor.",
     "info"),
    ("05", "Split sin leakage",
     "GroupShuffleSplit por case_id · 80/20 train/test",
     "El split se hace a nivel de caso quirúrgico (case_id) para evitar que latidos del mismo "
     "paciente aparezcan en train y test a la vez. Garantiza generalización real.",
     "info"),
    ("06", "Entrenamiento y búsqueda de HP",
     "RandomizedSearchCV · GroupKFold · 5 folds",
     "Se comparan: Logistic Regression, Decision Tree, Random Forest, SVM (LinearSVC), MLP. "
     "Búsqueda de hiperparámetros con RandomizedSearchCV y GroupKFold para mantener la "
     "restricción de case_id dentro del CV.",
     "info"),
    ("07", "Evaluación en test set",
     f"Métrica principal: F1-macro · Ganador: {winner} · F1={f1_str}",
     "Evaluación sobre el hold-out de test. F1-macro penaliza el fallo en clases minoritarias "
     "(p.ej. VT, SND). Se exportan: classification_report, confusion_matrix, feature_importance.",
     "teal"),
    ("08", "App Streamlit (esta demo)",
     "Predicción sobre datos tabulares · sin retrain",
     "La app carga el pipeline .joblib entrenado y permite evaluar casos VitalDB conocidos "
     "o explorar resultados por clase, features y matriz de confusión.",
     "muted"),
]

for step_num, step_name, step_sub, step_desc, step_accent in _STEPS:
    with st.container(border=True):
        col_n, col_c = st.columns([0.5, 9.5])
        with col_n:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:13px;font-weight:700;'
                f'color:var(--fg-4);background:var(--bg-3);border-radius:5px;'
                f'padding:4px 6px;text-align:center;margin-top:4px">{step_num}</div>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                f'<div style="font-size:14px;font-weight:600;color:var(--fg-0);'
                f'margin-bottom:2px">{step_name}</div>'
                f'<div style="font-size:11px;font-family:var(--mono);color:var(--fg-3);'
                f'margin-bottom:6px">{step_sub}</div>'
                f'<div style="font-size:12.5px;color:var(--fg-2);line-height:1.55">'
                f'{step_desc}</div>',
                unsafe_allow_html=True,
            )

st.write("")

# ── Decisiones de diseño ──────────────────────────────────────────────────────
section_title("Decisiones de diseño clave")

col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        card_header("¿Por qué features tabulares?", "enfoque metodológico")
        st.markdown(
            """Las señales ECG crudas requieren modelos 1D-CNN o transformers para aprendizaje
            de representaciones, lo que aumenta drásticamente el costo computacional y de datos.

            El enfoque tabular extrae **RR intervals + metadatos clínicos** (indicadores de riesgo
            ya validados en anestesiología) y los combina en un vector de features interpretable.
            Esto permite usar clasificadores clásicos de scikit-learn con tiempos de entrenamiento
            de segundos y explicabilidad directa por importancia de feature.""",
            unsafe_allow_html=False,
        )
        kv_table([
            ("Ventaja", "Interpretable, rápido, reproducible"),
            ("Limitación", "No captura morfología de la onda ECG"),
            ("Alternativa futura", "1D-CNN sobre ventanas de señal"),
        ])

with col_b:
    with st.container(border=True):
        card_header("¿Por qué F1-macro como métrica?", "evaluación multiclase desbalanceada")
        st.markdown(
            """El dataset está fuertemente desbalanceado: la clase Normal (N) tiene órdenes de
            magnitud más ejemplos que arritmias raras como VT o SND.

            **Accuracy** sería engañosa: un modelo que siempre predice N alcanzaría alta accuracy
            sin detectar ninguna arritmia real. **F1-macro** promedia el F1 de cada clase por igual,
            penalizando severamente el fallo en clases minoritarias.""",
            unsafe_allow_html=False,
        )
        kv_table([
            ("Accuracy del ganador", f"{meta.get('winner_metrics', {}).get('test_accuracy', 0):.3f}" if meta else "—"),
            ("F1-macro del ganador", f1_str),
            ("Diferencia", "Accuracy engañosa para clases raras"),
        ])

st.write("")

# ── Clases de ritmo ───────────────────────────────────────────────────────────
section_title("Clases de ritmo cardíaco")

_CLASES = [
    ("N", "Normal sinus rhythm", "Ritmo sinusal normal — clase mayoritaria (~60-70%)", "muted"),
    ("AFIB/AFL", "Atrial fibrillation / flutter", "Fibrilación o flutter auricular", "info"),
    ("Patterned Ventricular Ectopy", "PVE", "Ectopia ventricular con patrón definido", "warn"),
    ("Patterned Atrial Ectopy", "PAE", "Ectopia auricular con patrón definido", "warn"),
    ("SVTA", "Supraventricular tachyarrhythmia", "Taquiarritmia supraventricular", "warn"),
    ("VT", "Ventricular tachycardia", "Taquicardia ventricular — clase minoritaria crítica", "err"),
    ("SND", "Sinus node dysfunction", "Disfunción del nodo sinusal", "err"),
    ("AVB", "AV block", "Bloqueo auriculo-ventricular", "err"),
    ("WAP/MAT", "Wandering pacemaker / MAT", "Marcapasos errante o taquicardia auricular multifocal", "muted"),
    ("Unclassifiable", "—", "Latidos que no encajan en categorías anteriores", "muted"),
]

_cls_cols = st.columns(2)
for i, (code, full, desc, accent) in enumerate(_CLASES):
    with _cls_cols[i % 2]:
        st.markdown(
            f'<div style="display:flex;gap:8px;align-items:flex-start;'
            f'padding:7px 0;border-bottom:1px dashed var(--line-1)">'
            f'<div style="min-width:28px;margin-top:1px">{badge(code, accent)}</div>'
            f'<div>'
            f'<div style="font-size:12px;color:var(--fg-1);font-weight:500">{full}</div>'
            f'<div style="font-size:11px;color:var(--fg-3)">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

callout(
    "warn",
    "Limitaciones del modelo",
    "El modelo tabular no captura la morfología de la onda ECG — solo usa intervalos RR y "
    "metadatos clínicos. Las clases raras (VT, SND, AVB) tienen recall muy bajo por escasez "
    "de datos de entrenamiento. <b>No usar para diagnóstico clínico real.</b>",
)
