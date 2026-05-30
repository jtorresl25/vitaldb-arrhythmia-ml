"""Página de diagnóstico temporal — verifica el entorno de Streamlit Cloud.

Eliminar esta página una vez confirmado que el despliegue funciona correctamente.
"""

import importlib.util
import os
import sys

import streamlit as st

st.set_page_config(page_title="Diagnóstico", page_icon="🔧")
st.title("🔧 Diagnóstico de entorno")
st.caption("Página temporal para verificar dependencias en Streamlit Cloud.")

st.subheader("Python y sistema")
st.code(f"Python: {sys.version}", language=None)
st.code(f"Working directory: {os.getcwd()}", language=None)
st.code(f"sys.path[0]: {sys.path[0] if sys.path else '(vacío)'}", language=None)

st.subheader("Dependencias requeridas")

_PACKAGES = [
    "streamlit",
    "plotly",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "joblib",
    "pyarrow",
    "matplotlib",
    "PIL",
    "xgboost",
    "imblearn",
    "streamlit_option_menu",
]

rows = []
for pkg in _PACKAGES:
    spec = importlib.util.find_spec(pkg)
    status = "✅ OK" if spec else "❌ MISSING"
    try:
        mod = __import__(pkg.split(".")[0])
        version = getattr(mod, "__version__", "?")
    except Exception:
        version = "—"
    rows.append({"Paquete": pkg, "Estado": status, "Versión": version})

import pandas as _pd
st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Archivos clave del repo")

from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent.parent

_CHECK_PATHS = {
    "requirements.txt (raíz)":            _PROJECT_ROOT / "requirements.txt",
    "models/tabular_best_model_pipeline.joblib": _PROJECT_ROOT / "models" / "tabular_best_model_pipeline.joblib",
    "models/tabular_best_model_metadata.json":   _PROJECT_ROOT / "models" / "tabular_best_model_metadata.json",
    "data/processed/filtered_tabular_modeling_dataset.parquet": _PROJECT_ROOT / "data" / "processed" / "filtered_tabular_modeling_dataset.parquet",
}

path_rows = []
for label, path in _CHECK_PATHS.items():
    exists = path.exists()
    path_rows.append({
        "Archivo": label,
        "Estado": "✅ encontrado" if exists else "❌ no encontrado",
        "Ruta absoluta": str(path),
    })

st.dataframe(_pd.DataFrame(path_rows), use_container_width=True, hide_index=True)

st.subheader("requirements.txt cargado")
req_path = _PROJECT_ROOT / "requirements.txt"
if req_path.exists():
    st.code(req_path.read_text(encoding="utf-8"), language="text")
else:
    st.error("requirements.txt no encontrado desde la ruta resuelta.")
    st.write("Ruta buscada:", req_path)
