"""Benchmark de 5+ modelos sobre el dataset tabular filtrado.

Entrena y compara de forma reproducible los cinco modelos prometidos en la
metodología del proyecto:

    1. DummyClassifier   (baseline de piso)
    2. Decision Tree
    3. Linear SVC
    4. Random Forest
    5. XGBoost
    6. MLP
    (7. Logistic Regression, opcional)

Modos de ejecución:
    --fast    (default) max_cases=250, n_iter=5, n_splits=2, top_features=20
              Tiempo estimado: 8-15 min
    --medium  max_cases=400, n_iter=10, n_splits=3, top_features=25
              Tiempo estimado: 25-45 min
    --full    todos los casos, n_iter=20, n_splits=5, todas las features
              Tiempo estimado: 2-4 h (no recomendado en laptop)

Salidas producidas:
    reports/tables/model_comparison.csv
    reports/tables/best_model_classification_report.csv
    reports/tables/best_model_feature_importance.csv
    reports/tables/test_predictions.csv
    reports/tables/confusion_matrix.csv
    reports/tables/tabular_model_comparison_test.csv
    reports/tables/tabular_model_comparison_cv.csv
    models/best_model_pipeline.joblib
    models/model_artifacts_metadata.json
    models/feature_columns.json
    models/tabular_best_model_pipeline.joblib
    models/tabular_best_model_metadata.json

Uso:
    python scripts/04_run_benchmark_tabular.py            # modo fast
    python scripts/04_run_benchmark_tabular.py --medium
    python scripts/04_run_benchmark_tabular.py --full
    python scripts/04_run_benchmark_tabular.py --models decision_tree,random_forest --fast
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import RandomizedSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.modeling import make_train_test_group_split_with_coverage
from src.tabular_search import (
    PRIMARY_SCORING, SCORING_METRICS, TABULAR_MODEL_NAMES,
    build_cv_splitter, build_pipeline_for_model, classify_features,
    evaluate_on_test, extract_feature_importance, load_tabular_modeling_dataset,
    run_search_for_model,
)
from src.evaluation import class_support_per_split, per_class_report
from src.utils import ensure_dir, get_logger, set_seed

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ALL_BENCHMARK_MODELS = [
    "dummy",
    "decision_tree",
    "linear_svc",
    "random_forest",
    "xgboost",
    "mlp",
    "logreg",
]

# Distribuciones de hiperparámetros por modo --------------------------------
_PARAMS_FAST: dict[str, dict] = {
    "dummy":         {},
    "logreg":        {"clf__C": loguniform(1e-2, 10)},
    "decision_tree": {
        "clf__max_depth":        [None, 10, 20],
        "clf__min_samples_split":[2, 10, 50],
        "clf__min_samples_leaf": [1, 5, 20],
        "clf__criterion":        ["gini", "entropy"],
    },
    "linear_svc":    {"clf__C": loguniform(1e-2, 10)},
    "random_forest": {
        "clf__n_estimators":     [50, 100, 200],
        "clf__max_depth":        [None, 10, 20],
        "clf__min_samples_split":[2, 10],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_features":     ["sqrt", "log2"],
    },
    "xgboost": {
        "clf__n_estimators":  [50, 100, 200],
        "clf__max_depth":     [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample":     [0.8, 1.0],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "clf__alpha":              [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-3, 5e-3],
        "clf__activation":         ["relu", "tanh"],
    },
}

_PARAMS_MEDIUM: dict[str, dict] = {
    "dummy": {},
    "logreg":        {"clf__C": loguniform(1e-3, 50)},
    "decision_tree": {
        "clf__max_depth":        [None, 5, 10, 20, 30],
        "clf__min_samples_split":[2, 5, 10, 50, 100],
        "clf__min_samples_leaf": [1, 2, 5, 20],
        "clf__criterion":        ["gini", "entropy"],
    },
    "linear_svc":    {"clf__C": loguniform(1e-3, 1e2)},
    "random_forest": {
        "clf__n_estimators":     [100, 200, 300],
        "clf__max_depth":        [None, 10, 20, 30],
        "clf__min_samples_split":[2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features":     ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "clf__n_estimators":     [100, 200, 400],
        "clf__max_depth":        [3, 5, 7, 10],
        "clf__learning_rate":    [0.03, 0.05, 0.1, 0.2],
        "clf__subsample":        [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
        "clf__alpha":              [1e-5, 1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [5e-4, 1e-3, 5e-3],
        "clf__activation":         ["relu", "tanh"],
    },
}

# ---------------------------------------------------------------------------
# Pipeline factories específicas del benchmark
# ---------------------------------------------------------------------------

def _build_benchmark_pipeline(model_name: str, numeric: list[str], categorical: list[str],
                               fast: bool = True):
    """Pipeline con hiperparámetros por defecto ajustados para tiempos razonables."""
    from src.preprocessing import build_tabular_preprocessor
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    pre = build_tabular_preprocessor(
        numeric_features=numeric,
        categorical_features=categorical,
        with_scaling=True,
        ohe_min_frequency=config.TABULAR_OHE_MIN_FREQUENCY,
    )

    if model_name == "dummy":
        clf = DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_SEED)
    elif model_name == "logreg":
        clf = LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=300 if fast else 1000,
            random_state=config.RANDOM_SEED,
        )
    elif model_name == "decision_tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=config.RANDOM_SEED)
    elif model_name == "linear_svc":
        clf = LinearSVC(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            max_iter=1000 if fast else 3000,
            dual="auto",
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        from src.tabular_search import _XGBClassifierSafe
        clf = _XGBClassifierSafe(
            tree_method="hist",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )
    elif model_name == "mlp":
        clf = MLPClassifier(
            random_state=config.RANDOM_SEED,
            max_iter=80 if fast else 150,
            early_stopping=True,
            n_iter_no_change=7,
            validation_fraction=0.1,
        )
    else:
        raise KeyError(f"Modelo desconocido: {model_name!r}")

    return Pipeline(steps=[("preprocessor", pre), ("clf", clf)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_top_features(numeric, categorical, top_n, fi_path, logger):
    """Filtra numeric/categorical a las top_n columnas por importancia guardada."""
    if not fi_path.exists():
        logger.warning("--top-features %d: %s no existe. Usando todas.", top_n, fi_path)
        return numeric, categorical
    fi_df = pd.read_csv(fi_path)
    if "feature" not in fi_df.columns:
        logger.warning("--top-features: columna 'feature' no encontrada. Usando todas.")
        return numeric, categorical
    imp_col = (
        "abs_coef" if "abs_coef" in fi_df.columns
        else "importance" if "importance" in fi_df.columns
        else next((c for c in fi_df.columns if c not in ("feature","class")
                   and pd.api.types.is_numeric_dtype(fi_df[c])), None)
    )
    if imp_col is None:
        return numeric, categorical

    num_set = set(numeric); cat_set = set(categorical)

    def _orig(feat):
        if feat.startswith("num__"):
            c = feat[5:]; return c if c in num_set else None
        if feat.startswith("cat__"):
            rem = feat[5:]
            best = None
            for col in categorical:
                if (rem == col or rem.startswith(col + "_")) and (best is None or len(col) > len(best)):
                    best = col
            return best
        return feat if (feat in num_set or feat in cat_set) else None

    col_imp: dict[str, float] = {}
    for _, row in fi_df.iterrows():
        orig = _orig(str(row["feature"]))
        if orig is None: continue
        col_imp[orig] = col_imp.get(orig, 0.0) + abs(float(row[imp_col]) if pd.notna(row[imp_col]) else 0.0)

    if not col_imp:
        return numeric, categorical

    top_cols = set(sorted(col_imp, key=col_imp.get, reverse=True)[:top_n])
    new_num = [f for f in numeric if f in top_cols]
    new_cat = [f for f in categorical if f in top_cols]
    n_got = len(new_num) + len(new_cat)
    logger.info("--top-features %d: %d seleccionadas (num=%d cat=%d). Top-5: %s",
                top_n, n_got, len(new_num), len(new_cat),
                sorted(top_cols, key=col_imp.get, reverse=True)[:5])
    return (new_num, new_cat) if n_got > 0 else (numeric, categorical)


def _subsample_cases(df, max_cases, random_state):
    rng = np.random.RandomState(random_state)
    ids = df[config.CASE_ID_COLUMN].drop_duplicates().to_numpy()
    if max_cases >= len(ids): return df
    chosen = rng.choice(ids, size=max_cases, replace=False)
    return df.loc[df[config.CASE_ID_COLUMN].isin(chosen)].copy()


def _run_dummy(X_train, y_train, X_test, y_test, numeric, categorical, random_state):
    """Entrena DummyClassifier y devuelve dict de métricas."""
    from src.preprocessing import build_tabular_preprocessor
    from sklearn.pipeline import Pipeline
    pre = build_tabular_preprocessor(
        numeric_features=numeric, categorical_features=categorical,
        with_scaling=False, ohe_min_frequency=config.TABULAR_OHE_MIN_FREQUENCY,
    )
    pipe = Pipeline([("preprocessor", pre),
                     ("clf", DummyClassifier(strategy="most_frequent",
                                             random_state=random_state))])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    elapsed = time.time() - t0
    row = {
        "model": "dummy",
        "status": "ok",
        "fit_time_seconds": elapsed,
        "cv_f1_macro": 0.0,
        "test_f1_macro":         float(f1_score(y_test, y_pred, average="macro",     zero_division=0)),
        "test_precision_macro":  float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_recall_macro":     float(recall_score(y_test, y_pred, average="macro",  zero_division=0)),
        "test_accuracy":         float(accuracy_score(y_test, y_pred)),
        "test_f1_weighted":      float(f1_score(y_test, y_pred, average="weighted",   zero_division=0)),
        "best_params": "{}",
        "n_iter_effective": 0,
        "error_message": "",
    }
    return row, pipe, y_pred


def _save_partial_comparison(model_rows, tables_dir, models_dir, numeric, categorical, logger):
    """Persiste tablas + mejor modelo disponibles hasta el momento."""
    df = pd.DataFrame(model_rows)
    df.to_csv(tables_dir / "model_comparison.csv", index=False)
    df.to_csv(tables_dir / "tabular_model_comparison_test.csv", index=False)

    ok = df.loc[df["status"] == "ok"].dropna(subset=["test_f1_macro"])
    if ok.empty: return
    winner = ok.sort_values("test_f1_macro", ascending=False).iloc[0].to_dict()
    m = winner["model"]
    logger.info("[partial] mejor hasta ahora: %s (test_f1_macro=%.3f)", m, winner["test_f1_macro"])


def _plot_confusion(y_true, y_pred, title, out_path, normalize=False):
    labels = sorted(set(pd.Series(y_true).unique()) | set(pd.Series(y_pred).unique()), key=str)
    if normalize:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true"); fmt = ".2f"
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels); fmt = "d"
    fig, ax = plt.subplots(figsize=(1.0 + 0.9*len(labels), 1.0 + 0.7*len(labels)))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Parseo de argumentos
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=None,
                   help=f"Modelos separados por coma. Disponibles: {','.join(ALL_BENCHMARK_MODELS)}. "
                        "Default: todos excepto logreg.")
    p.add_argument("--fast",   action="store_true", help="Modo rápido (default si ningún modo indicado).")
    p.add_argument("--medium", action="store_true", help="Modo intermedio.")
    p.add_argument("--full",   action="store_true", help="Modo completo (lento).")
    p.add_argument("--max-cases",    type=int,   default=None)
    p.add_argument("--n-iter",       type=int,   default=None)
    p.add_argument("--n-splits",     type=int,   default=None)
    p.add_argument("--top-features", type=int,   default=None,
                   help="Top N features por importancia guardada. Default según modo.")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--random-state", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()
    logger = get_logger("benchmark")
    set_seed(args.random_state)

    # Determinar modo
    if args.full:
        mode = "full";   max_cases_def = None; n_iter_def = 20; n_splits_def = 5; top_feat_def = None
    elif args.medium:
        mode = "medium"; max_cases_def = 400;  n_iter_def = 10; n_splits_def = 3; top_feat_def = 25
    else:
        mode = "fast";   max_cases_def = 250;  n_iter_def = 5;  n_splits_def = 2; top_feat_def = 20

    max_cases  = args.max_cases    or max_cases_def
    n_iter     = args.n_iter       or n_iter_def
    n_splits   = args.n_splits     or n_splits_def
    top_feat   = args.top_features or top_feat_def
    param_dist = _PARAMS_MEDIUM if mode == "medium" else (_PARAMS_FAST if mode == "fast" else _PARAMS_MEDIUM)

    logger.info("Modo: %s | max_cases=%s | n_iter=%d | n_splits=%d | top_features=%s",
                mode, max_cases, n_iter, n_splits, top_feat)

    # Modelos a ejecutar
    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in models_to_run if m not in ALL_BENCHMARK_MODELS]
        if unknown:
            raise SystemExit(f"Modelos desconocidos: {unknown}")
    else:
        models_to_run = [m for m in ALL_BENCHMARK_MODELS if m != "logreg"]

    # Dirs
    tables_dir  = ensure_dir(PROJECT_ROOT / "reports" / "tables")
    figures_dir = ensure_dir(PROJECT_ROOT / "reports" / "figures")
    models_dir  = ensure_dir(PROJECT_ROOT / "models")

    # ------------------------------------------------------------------
    # 1. Carga y preparación de datos
    # ------------------------------------------------------------------
    df = load_tabular_modeling_dataset()
    logger.info("Dataset shape=%s cases=%d classes=%d",
                df.shape, df[config.CASE_ID_COLUMN].nunique(), df[config.TARGET_COLUMN].nunique())

    if max_cases is not None:
        df = _subsample_cases(df, max_cases, args.random_state)
        logger.info("Tras subsample: shape=%s cases=%d", df.shape, df[config.CASE_ID_COLUMN].nunique())

    cls       = classify_features(df)
    numeric   = cls["numeric_features"]
    categorical = cls["categorical_features"]
    logger.info("Features: %d numeric + %d categorical", len(numeric), len(categorical))

    if top_feat is not None:
        fi_path = tables_dir / "tabular_feature_importance_best_model.csv"
        numeric, categorical = _select_top_features(numeric, categorical, top_feat, fi_path, logger)
        logger.info("Tras top-%d: %d + %d features", top_feat, len(numeric), len(categorical))

    # Persistir feature list
    pd.DataFrame({
        "feature": numeric + categorical,
        "kind": ["numeric"]*len(numeric) + ["categorical"]*len(categorical),
    }).to_csv(tables_dir / "tabular_feature_list_used.csv", index=False)
    # Legacy feature_columns.json (lista plana)
    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(numeric + categorical, f, indent=2)

    # ------------------------------------------------------------------
    # 2. Split por case_id
    # ------------------------------------------------------------------
    X_df = df[numeric + categorical]
    y    = df[config.TARGET_COLUMN].to_numpy()
    groups = df[config.CASE_ID_COLUMN].to_numpy()

    train_idx, test_idx, split_info = make_train_test_group_split_with_coverage(
        X_df, y, groups, test_size=0.2, random_state=args.random_state, max_attempts=200,
    )
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "Fuga de grupo."

    n_train_groups = len(set(groups[train_idx].tolist()))
    n_test_groups  = len(set(groups[test_idx].tolist()))
    logger.info("Split: train=%d rows / %d grupos | test=%d rows / %d grupos",
                len(train_idx), n_train_groups, len(test_idx), n_test_groups)

    support_df = class_support_per_split(y_train, y_test)
    support_df.reset_index(names="class").to_csv(
        tables_dir / "tabular_class_support_train_test.csv", index=False)

    # Codificar target en enteros para compatibilidad con MLP (early_stopping)
    # y cualquier otro modelo que no tolere labels strings con np.isnan.
    # Los reportes/predicciones finales usan siempre las etiquetas originales.
    from sklearn.preprocessing import LabelEncoder
    _le = LabelEncoder().fit(np.unique(y))
    y_train_enc = _le.transform(y_train)
    y_test_enc  = _le.transform(y_test)

    # ------------------------------------------------------------------
    # 3. CV builder
    # ------------------------------------------------------------------
    cv, cv_name, n_splits_eff = build_cv_splitter(groups_train, y_train, n_splits)
    logger.info("CV: %s | n_splits_efectivo=%d", cv_name, n_splits_eff)

    # ------------------------------------------------------------------
    # 4. Entrenamiento de todos los modelos
    # ------------------------------------------------------------------
    model_rows:    list[dict]    = []
    fitted_models: dict[str, object] = {}
    y_preds:       dict[str, np.ndarray] = {}

    for model_name in models_to_run:
        logger.info("=" * 60)
        logger.info("Modelo: %s  [modo=%s n_iter=%d]", model_name.upper(), mode, n_iter)
        t0 = time.time()

        # DummyClassifier: sin RandomizedSearchCV
        if model_name == "dummy":
            try:
                row, pipe, y_pred = _run_dummy(
                    X_train, y_train, X_test, y_test, numeric, categorical, args.random_state)
                fitted_models["dummy"] = pipe
                y_preds["dummy"]       = y_pred
                logger.info("  dummy: test_f1_macro=%.3f  fit=%.1fs",
                            row["test_f1_macro"], row["fit_time_seconds"])
            except Exception as exc:
                row = {"model": "dummy", "status": "error", "fit_time_seconds": time.time()-t0,
                       "error_message": str(exc), **{k: float("nan") for k in [
                           "cv_f1_macro","test_f1_macro","test_precision_macro",
                           "test_recall_macro","test_accuracy","test_f1_weighted"]},
                       "best_params": "", "n_iter_effective": 0}
                logger.error("  dummy ERROR: %s", exc)
            model_rows.append(row)
            _save_partial_comparison(model_rows, tables_dir, models_dir, numeric, categorical, logger)
            continue

        # Verificar disponibilidad de XGBoost
        if model_name == "xgboost":
            try:
                import xgboost  # noqa: F401
            except ImportError:
                logger.warning("XGBoost no instalado. Saltando.")
                model_rows.append({"model": "xgboost", "status": "error",
                                   "error_message": "xgboost not installed", **{
                                       k: float("nan") for k in ["cv_f1_macro","test_f1_macro",
                                       "test_precision_macro","test_recall_macro",
                                       "test_accuracy","test_f1_weighted"]},
                                   "fit_time_seconds": 0, "best_params": "", "n_iter_effective": 0})
                _save_partial_comparison(model_rows, tables_dir, models_dir, numeric, categorical, logger)
                continue

        try:
            # Construir pipeline con defaults ajustados para el modo
            pipe = _build_benchmark_pipeline(model_name, numeric, categorical, fast=(mode == "fast"))

            # Seleccionar distribución de parámetros
            pdist = param_dist.get(model_name, {})

            # DummyClassifier no necesita search
            if not pdist:
                # Sin búsqueda
                t1 = time.time()
                pipe.fit(X_train, y_train_enc)
                y_pred_enc = pipe.predict(X_test)
                elapsed = time.time() - t1
                cv_score = 0.0
                best_params = {}
                n_iter_eff = 1
            else:
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=pdist,
                    n_iter=n_iter,
                    scoring=SCORING_METRICS,
                    refit=PRIMARY_SCORING,
                    cv=cv,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    error_score=np.nan,
                    return_train_score=False,
                    verbose=0,
                )
                t1 = time.time()
                search.fit(X_train, y_train_enc, groups=groups_train)
                elapsed = time.time() - t1
                best_pipe  = search.best_estimator_
                y_pred_enc = best_pipe.predict(X_test)
                pipe       = best_pipe
                cv_score   = float(search.best_score_)
                best_params    = search.best_params_
                n_iter_eff     = len(search.cv_results_["params"])

            # Decodificar a labels originales para todos los reportes
            y_pred = _le.inverse_transform(np.asarray(y_pred_enc, dtype=int))

            fitted_models[model_name] = pipe
            y_preds[model_name]       = y_pred

            row = {
                "model":               model_name,
                "status":              "ok",
                "fit_time_seconds":    elapsed,
                "cv_f1_macro":         cv_score,
                "test_f1_macro":       float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
                "test_precision_macro":float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "test_recall_macro":   float(recall_score(y_test, y_pred, average="macro",  zero_division=0)),
                "test_accuracy":       float(accuracy_score(y_test, y_pred)),
                "test_f1_weighted":    float(f1_score(y_test, y_pred, average="weighted",   zero_division=0)),
                "best_params":         json.dumps(best_params, default=str),
                "n_iter_effective":    n_iter_eff,
                "error_message":       "",
            }
            logger.info("  %s OK | cv_f1=%.3f  test_f1=%.3f  acc=%.3f  fit=%.1fs",
                        model_name, cv_score, row["test_f1_macro"],
                        row["test_accuracy"], elapsed)

        except Exception as exc:
            logger.error("  %s ERROR: %s", model_name, exc)
            row = {
                "model": model_name, "status": "error",
                "fit_time_seconds": time.time() - t0, "error_message": str(exc),
                "cv_f1_macro": float("nan"),
                **{k: float("nan") for k in ["test_f1_macro","test_precision_macro",
                                              "test_recall_macro","test_accuracy","test_f1_weighted"]},
                "best_params": "", "n_iter_effective": 0,
            }

        model_rows.append(row)
        _save_partial_comparison(model_rows, tables_dir, models_dir, numeric, categorical, logger)

    # ------------------------------------------------------------------
    # 5. Identificar ganador
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(model_rows)
    ok_df = results_df.loc[results_df["status"] == "ok"].dropna(subset=["test_f1_macro"])

    # Dummy es solo baseline comparativo; nunca puede ser el modelo desplegable.
    ok_real = ok_df.loc[ok_df["model"] != "dummy"]

    if ok_real.empty:
        if ok_df.empty:
            logger.error("Ningún modelo completó con éxito (incluyendo dummy).")
        else:
            logger.error(
                "Solo DummyClassifier completó. Ningún modelo real (decision_tree, "
                "linear_svc, random_forest, xgboost, mlp, logreg) pudo entrenarse. "
                "No se guarda best_model_pipeline.joblib."
            )
        return 1

    winner_row  = ok_real.sort_values("test_f1_macro", ascending=False).iloc[0]
    winner_name = str(winner_row["model"])
    best_est    = fitted_models.get(winner_name)
    y_pred_best = y_preds.get(winner_name)

    logger.info("=" * 60)
    logger.info("GANADOR: %s | test_f1_macro=%.3f | test_accuracy=%.3f",
                winner_name, winner_row["test_f1_macro"], winner_row["test_accuracy"])

    # ------------------------------------------------------------------
    # 6. Añadir columnas de contexto a model_comparison.csv
    # ------------------------------------------------------------------
    n_feats = len(numeric) + len(categorical)
    results_df["n_features"]    = n_feats
    results_df["train_size"]    = len(train_idx)
    results_df["test_size"]     = len(test_idx)
    results_df["n_train_groups"]= n_train_groups
    results_df["n_test_groups"] = n_test_groups

    results_df.to_csv(tables_dir / "model_comparison.csv", index=False)
    results_df.to_csv(tables_dir / "tabular_model_comparison_test.csv", index=False)

    # CV view
    cv_cols = ["model","status","n_iter_effective","fit_time_seconds","cv_f1_macro","best_params"]
    results_df.reindex(columns=cv_cols).to_csv(
        tables_dir / "tabular_model_comparison_cv.csv", index=False)

    logger.info("Tablas comparativas guardadas.")

    # ------------------------------------------------------------------
    # 7. Artefactos del modelo ganador
    # ------------------------------------------------------------------
    if best_est is not None and y_pred_best is not None:

        # Classification report (índice = clase o resumen)
        rep_dict = classification_report(y_test, y_pred_best, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep_dict).T
        rep_df.index.name = "class_or_avg"
        rep_df.to_csv(tables_dir / "best_model_classification_report.csv")
        rep_df.to_csv(tables_dir / "tabular_best_model_classification_report.csv")
        logger.info("Classification report guardado.")

        # Feature importance
        fi = extract_feature_importance(best_est)
        if fi is not None:
            fi.to_csv(tables_dir / "best_model_feature_importance.csv",    index=False)
            fi.to_csv(tables_dir / "tabular_feature_importance_best_model.csv", index=False)
            logger.info("Feature importance guardada (%d filas).", len(fi))
        else:
            pd.DataFrame([{"note": f"No se pudo extraer importancia para {winner_name}."}]).to_csv(
                tables_dir / "best_model_feature_importance.csv", index=False)

        # test_predictions.csv
        time_col  = "time_second" if "time_second" in df.columns else None
        case_col  = config.CASE_ID_COLUMN if config.CASE_ID_COLUMN in df.columns else None
        pred_rows = {}
        if time_col:  pred_rows[time_col]        = df.iloc[test_idx][time_col].values
        if case_col:  pred_rows[case_col]         = df.iloc[test_idx][case_col].values
        pred_rows["rhythm_label"]                 = y_test
        pred_rows["prediction"]                   = y_pred_best
        pred_rows["correct"]                      = (y_test == y_pred_best).astype(int)
        pd.DataFrame(pred_rows).to_csv(tables_dir / "test_predictions.csv", index=False)
        logger.info("test_predictions.csv guardado (%d filas).", len(test_idx))

        # confusion_matrix.csv (formato long para el viewer interactivo)
        labels_all = sorted(
            set(pd.Series(y_test).unique()) | set(pd.Series(y_pred_best).unique()), key=str
        )
        cm = confusion_matrix(y_test, y_pred_best, labels=labels_all)
        cm_long = [
            {"real_label": labels_all[i], "predicted_label": labels_all[j], "count": int(cm[i, j])}
            for i in range(len(labels_all)) for j in range(len(labels_all))
        ]
        cm_df = pd.DataFrame(cm_long)
        cm_df.to_csv(tables_dir / "confusion_matrix.csv", index=False)
        logger.info("confusion_matrix.csv guardado.")

        # Matriz de confusión PNG
        _plot_confusion(y_test, y_pred_best,
                        f"Matriz de confusión — {winner_name}",
                        figures_dir / "best_model_confusion_matrix.png")
        _plot_confusion(y_test, y_pred_best,
                        f"Matriz de confusión — {winner_name}",
                        figures_dir / "tabular_best_model_confusion_matrix_absolute.png")

        # Guardar modelos (ambas convenciones de nombre)
        joblib.dump(best_est, models_dir / "best_model_pipeline.joblib")
        joblib.dump(best_est, models_dir / "tabular_best_model_pipeline.joblib")
        logger.info("Modelos guardados: best_model_pipeline.joblib + tabular_best_model_pipeline.joblib")

        # models/model_artifacts_metadata.json (formato legacy completo)
        best_params_per_model = {}
        for mname in models_to_run:
            bp_row = results_df.loc[results_df["model"] == mname]
            if not bp_row.empty and str(bp_row["status"].iloc[0]) == "ok":
                try:
                    hp = json.loads(str(bp_row["best_params"].iloc[0]))
                    best_params_per_model[mname] = hp
                except Exception:
                    pass

        import sklearn, numpy as np_import
        try:
            import xgboost as _xgb; xgb_ver = _xgb.__version__
        except ImportError:
            xgb_ver = "not installed"
        try:
            import joblib as _jl; jl_ver = _jl.__version__
        except Exception:
            jl_ver = "unknown"

        meta_legacy = {
            "training_datetime":        datetime.datetime.now().isoformat(),
            "benchmark_mode":           mode,
            "winner_model":             winner_name,
            "primary_metric":           "test_f1_macro",
            "winner_test_f1_macro":     float(winner_row["test_f1_macro"]),
            "winner_test_accuracy":     float(winner_row["test_accuracy"]),
            "winner_test_precision":    float(winner_row["test_precision_macro"]),
            "winner_test_recall":       float(winner_row["test_recall_macro"]),
            "target":                   config.TARGET_COLUMN,
            "forbidden_columns":        list(config.TABULAR_LEAKAGE_COLUMNS),
            "feature_cols":             numeric + categorical,
            "n_features":               n_feats,
            "train_size_rows":          int(len(train_idx)),
            "test_size_rows":           int(len(test_idx)),
            "n_train_groups":           n_train_groups,
            "n_test_groups":            n_test_groups,
            "models_compared":          models_to_run,
            "best_hyperparams_per_model": best_params_per_model,
            "library_versions": {
                "sklearn": sklearn.__version__,
                "numpy":   np_import.__version__,
                "pandas":  pd.__version__,
                "xgboost": xgb_ver,
                "joblib":  jl_ver,
            },
        }
        with open(models_dir / "model_artifacts_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_legacy, f, ensure_ascii=False, indent=2, default=str)

        # models/tabular_best_model_metadata.json (formato nuevo)
        meta_tabular = {
            "winner_model":         winner_name,
            "winner_test_f1_macro": float(winner_row["test_f1_macro"]),
            "target":               config.TARGET_COLUMN,
            "numeric_features":     numeric,
            "categorical_features": categorical,
            "trained_at":           datetime.datetime.now().isoformat(),
            "best_params":          str(winner_row.get("best_params", "{}")),
            "winner_metrics": {
                k: float(winner_row[k])
                for k in ["test_f1_macro","test_precision_macro","test_recall_macro",
                          "test_accuracy","test_f1_weighted","cv_f1_macro"]
                if k in winner_row and pd.notna(winner_row[k])
            },
            "benchmark_mode": mode,
        }
        with open(models_dir / "tabular_best_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_tabular, f, ensure_ascii=False, indent=2, default=str)

        logger.info("Metadata guardada: model_artifacts_metadata.json + tabular_best_model_metadata.json")

    # ------------------------------------------------------------------
    # 8. Resumen final
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("RESUMEN FINAL:")
    for _, row in results_df.iterrows():
        if str(row["status"]) == "ok":
            logger.info(
                "  %-15s | f1_macro=%.3f | accuracy=%.3f | time=%.0fs",
                row["model"], row["test_f1_macro"], row["test_accuracy"], row["fit_time_seconds"],
            )
        else:
            logger.info("  %-15s | ERROR: %s", row["model"], str(row.get("error_message",""))[:60])
    logger.info("Ganador: %s (test_f1_macro=%.3f)", winner_name, winner_row["test_f1_macro"])
    logger.info("Salidas guardadas en reports/tables/ y models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
