import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.paths import (
    DATA_DIR,
    MODELS_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
    ARTIFACTS_MODELS_DIR,
    ARTIFACTS_TABLES_DIR,
    ARTIFACTS_FIGURES_DIR,
    resolve_path,
)

TARGET_COLUMN = "rhythm_label"


def file_exists(path: Path) -> bool:
    return Path(path).exists()


# ---------------------------------------------------------------------------
# Internal helper: check app_artifacts first, then project-root fallback
# ---------------------------------------------------------------------------
def _pick(artifacts_dir: Path, fallback_dir: Path, filename: str) -> "Path | None":
    """Return the first existing path: artifacts_dir/filename, else fallback_dir/filename."""
    return resolve_path(artifacts_dir / filename, fallback_dir / filename)


# ---------------------------------------------------------------------------
# Normalizer: maps tabular metadata keys → superset compatible with legacy
# ---------------------------------------------------------------------------
def _normalize_metadata(raw: dict) -> dict:
    out = dict(raw)
    # trained_at → training_datetime
    if "trained_at" in raw and "training_datetime" not in raw:
        out["training_datetime"] = raw["trained_at"]
    # numeric + categorical lists → n_features scalar
    if "n_features" not in out:
        n = len(raw.get("numeric_features", [])) + len(raw.get("categorical_features", []))
        if n:
            out["n_features"] = n
    # best_params JSON string → best_hyperparams_per_model dict
    if "best_hyperparams_per_model" not in out:
        winner = raw.get("winner_model", "")
        bp_str = raw.get("best_params", "")
        if winner and bp_str:
            try:
                hp = json.loads(bp_str)
                hp_clean = {
                    k.replace("clf__", "").replace("preprocessor__", ""): v
                    for k, v in hp.items()
                }
                out["best_hyperparams_per_model"] = {winner: hp_clean}
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@st.cache_data
def load_model_metadata() -> dict | None:
    """Carga metadata del modelo. Busca en app_artifacts primero; fallback a models/."""
    candidates = [
        (ARTIFACTS_MODELS_DIR / "tabular_best_model_metadata.json", True),
        (MODELS_DIR / "tabular_best_model_metadata.json", True),
        (MODELS_DIR / "model_artifacts_metadata.json", False),
    ]
    for path, is_tabular in candidates:
        if file_exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return _normalize_metadata(raw) if is_tabular else raw
    return None


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------
@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    """Carga comparativa de modelos. app_artifacts primero; fallback legacy."""
    test_path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "tabular_model_comparison_test.csv")
    cv_path   = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "tabular_model_comparison_cv.csv")
    if test_path is not None:
        df_test = pd.read_csv(test_path)
        if cv_path is not None:
            df_cv = pd.read_csv(cv_path)
            extra = ["model"] + [c for c in df_cv.columns if c not in df_test.columns]
            df_test = df_test.merge(df_cv[extra], on="model", how="left")
        return df_test
    # Legacy fallback
    legacy = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "model_comparison.csv")
    return pd.read_csv(legacy) if legacy else None


# ---------------------------------------------------------------------------
# Classification report (per-class)
# ---------------------------------------------------------------------------
@st.cache_data
def load_classification_report() -> pd.DataFrame | None:
    for fname in (
        "tabular_best_model_classification_report.csv",
        "best_model_classification_report.csv",
    ):
        path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, fname)
        if path is not None:
            return pd.read_csv(path, index_col=0)
    return None


# ---------------------------------------------------------------------------
# Train/test split summary (metric,value CSV)
# ---------------------------------------------------------------------------
@st.cache_data
def load_train_test_split_summary() -> dict | None:
    """Devuelve tabular_train_test_split_summary.csv como dict {metric: value}."""
    path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "tabular_train_test_split_summary.csv")
    if path is None:
        return None
    df = pd.read_csv(path)
    return dict(zip(df["metric"].astype(str), df["value"]))


# ---------------------------------------------------------------------------
# Class support per split (class, train, test, total)
# ---------------------------------------------------------------------------
@st.cache_data
def load_class_support_train_test() -> "pd.DataFrame | None":
    """Devuelve tabular_class_support_train_test.csv con soporte por clase."""
    path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "tabular_class_support_train_test.csv")
    return pd.read_csv(path) if path else None


# ---------------------------------------------------------------------------
# Historical model comparison (exploratory runs only)
# ---------------------------------------------------------------------------
@st.cache_data
def load_model_comparison_history() -> pd.DataFrame | None:
    """Carga histórico de benchmarks exploratorios (5 modelos, 150 casos)."""
    path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "tabular_model_comparison_history.csv")
    return pd.read_csv(path) if path else None


# ---------------------------------------------------------------------------
# Binary per-case evaluation metrics (output of 05_select_binary_demo_cases.py)
# ---------------------------------------------------------------------------
@st.cache_data
def load_binary_case_level_metrics() -> "pd.DataFrame | None":
    """Carga binary_case_level_metrics.csv — métricas por case_id del dataset completo."""
    path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "binary_case_level_metrics.csv")
    return pd.read_csv(path) if path else None


# ---------------------------------------------------------------------------
# Final official model record
# ---------------------------------------------------------------------------
@st.cache_data
def load_model_final_official() -> pd.DataFrame | None:
    """Carga el registro del modelo final oficial (Linear SVC, dataset completo)."""
    path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "tabular_model_final_official.csv")
    return pd.read_csv(path) if path else None


# ---------------------------------------------------------------------------
# Binary metrics table
# ---------------------------------------------------------------------------
@st.cache_data
def load_binary_metrics() -> pd.DataFrame | None:
    path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, "tabular_binary_metrics.csv")
    return pd.read_csv(path) if path else None


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    for fname in (
        "tabular_feature_importance_best_model.csv",
        "best_model_feature_importance.csv",
    ):
        path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, fname)
        if path is not None:
            return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Feature column list
# ---------------------------------------------------------------------------
@st.cache_data
def load_feature_columns() -> list | None:
    """Devuelve la lista de features del modelo activo."""
    path_meta = _pick(ARTIFACTS_MODELS_DIR, MODELS_DIR, "tabular_best_model_metadata.json")
    if path_meta is not None:
        with open(path_meta, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        numeric     = raw.get("numeric_features", [])
        categorical = raw.get("categorical_features", [])
        if numeric or categorical:
            return numeric + categorical
    # Legacy fallback
    path_cols = _pick(ARTIFACTS_MODELS_DIR, MODELS_DIR, "feature_columns.json")
    if path_cols is not None:
        with open(path_cols, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Trained pipeline (joblib)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    for fname in ("tabular_best_model_pipeline.joblib", "best_model_pipeline.joblib"):
        path = _pick(ARTIFACTS_MODELS_DIR, MODELS_DIR, fname)
        if path is not None:
            try:
                import joblib
                return joblib.load(path)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Tabular parquet for prediction demo
# ---------------------------------------------------------------------------
@st.cache_data
def load_tabular_parquet(n_sample: int = 300) -> "pd.DataFrame | None":
    """Carga sample del dataset tabular procesado para la demo de predicción."""
    path = DATA_DIR / "processed" / "filtered_tabular_modeling_dataset.parquet"
    if not file_exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def confusion_matrix_figure_path() -> Path | None:
    for fname in (
        "tabular_best_model_confusion_matrix_absolute.png",
        "best_model_confusion_matrix.png",
    ):
        path = _pick(ARTIFACTS_FIGURES_DIR, REPORT_FIGURES_DIR, fname)
        if path is not None:
            return path
    return None


@st.cache_data
def load_confusion_matrix_csv() -> "pd.DataFrame | None":
    # Prefer long format (confusion_matrix.csv) — unambiguous column names.
    # Fall back to wide format (tabular_confusion_matrix_absolute.csv).
    for fname in (
        "confusion_matrix.csv",
        "tabular_confusion_matrix_absolute.csv",
    ):
        path = _pick(ARTIFACTS_TABLES_DIR, REPORT_TABLES_DIR, fname)
        if path is not None:
            return pd.read_csv(path)
    return None


def correlation_figure_path() -> Path | None:
    path = _pick(ARTIFACTS_FIGURES_DIR, REPORT_FIGURES_DIR, "feature_correlation_heatmap.png")
    return path
