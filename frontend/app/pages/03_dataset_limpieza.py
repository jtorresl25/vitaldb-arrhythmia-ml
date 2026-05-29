import streamlit as st

from components.layout import placeholder_page

placeholder_page(
    title="Dataset y limpieza",
    description=(
        "Auditoría de calidad sobre la VitalDB Arrhythmia Database. "
        "Distribución de clases de ritmo, cobertura de cases por clase, "
        "estadísticas de calidad de señal y balance del dataset tabular."
    ),
    files_needed=[
        "reports/tables/tabular_best_model_classification_report.csv  (distribución de clases)",
        "reports/tables/tabular_class_support_train_test.csv  (soporte por clase en split)",
    ],
    note="La distribución de clases se deriva del classification report del modelo tabular.",
)
