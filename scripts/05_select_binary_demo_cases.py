"""05_select_binary_demo_cases.py

Evalúa todos los case_id del dataset tabular con el modelo binario final
(Linear SVC), calcula métricas por caso y selecciona los mejores candidatos
para la demo en la app Streamlit.

Ejecutar desde la raíz del proyecto:
    python scripts/05_select_binary_demo_cases.py

Salidas:
    reports/tables/binary_case_level_metrics.csv    — métricas de todos los casos
    reports/tables/binary_demo_case_candidates.csv  — top candidatos por categoría
    frontend/app/app_artifacts/demo/demo_cases_binary.csv — catálogo final de la app
    frontend/app/app_artifacts/demo/npy_cases/      — .npy copiados si existen
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PARQUET_PATH  = PROJECT_ROOT / "data" / "processed" / "filtered_tabular_modeling_dataset.parquet"
MODEL_PATH    = PROJECT_ROOT / "models" / "tabular_best_model_pipeline.joblib"
META_PATH     = PROJECT_ROOT / "models" / "tabular_best_model_metadata.json"
DEMO_NPY_DIR  = PROJECT_ROOT / "data" / "demo" / "npy_cases"

REPORT_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
ARTIFACTS_DEMO_DIR    = PROJECT_ROOT / "frontend" / "app" / "app_artifacts" / "demo"
ARTIFACTS_NPY_DIR     = ARTIFACTS_DEMO_DIR / "npy_cases"
ARTIFACTS_TABLES_DIR  = PROJECT_ROOT / "frontend" / "app" / "app_artifacts" / "reports" / "tables"

OUT_METRICS    = REPORT_TABLES_DIR / "binary_case_level_metrics.csv"
OUT_CANDIDATES = REPORT_TABLES_DIR / "binary_demo_case_candidates.csv"
OUT_DEMO_CSV   = ARTIFACTS_DEMO_DIR / "demo_cases_binary.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_pred(arr) -> np.ndarray:
    """Normalise model output to lowercase 'normal' / 'abnormal' strings."""
    out = []
    for v in arr:
        s = str(v).lower().strip()
        if s in ("normal", "0", "n", "false"):
            out.append("normal")
        else:
            out.append("abnormal")
    return np.array(out)


def _case_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return per-case metric dict. y_true and y_pred are 1-D string arrays."""
    n = len(y_true)
    n_normal   = int((y_true == "normal").sum())
    n_abnormal = int((y_true == "abnormal").sum())
    has_both   = n_normal > 0 and n_abnormal > 0
    dominant   = "normal" if n_normal >= n_abnormal else "abnormal"

    acc  = accuracy_score(y_true, y_pred)
    n_err = int((y_true != y_pred).sum())

    # Balanced accuracy only meaningful with both classes
    bal_acc = balanced_accuracy_score(y_true, y_pred) if has_both else float("nan")

    # Per-class precision / recall / F1 for "abnormal" (positive class)
    labels = ["normal", "abnormal"]
    if has_both:
        prec_ab   = precision_score(y_true, y_pred, pos_label="abnormal",
                                    zero_division=0)
        rec_ab    = recall_score(y_true, y_pred, pos_label="abnormal",
                                 zero_division=0)
        f1_ab     = f1_score(y_true, y_pred, pos_label="abnormal",
                             zero_division=0)
        spec_norm = recall_score(y_true, y_pred, pos_label="normal",
                                 zero_division=0)
        f1_mac    = f1_score(y_true, y_pred, average="macro",
                             labels=labels, zero_division=0)
    elif n_abnormal == 0:
        # All normal
        prec_ab = rec_ab = f1_ab = float("nan")
        spec_norm = accuracy_score(y_true, y_pred)
        f1_mac    = float("nan")
    else:
        # All abnormal
        prec_ab   = precision_score(y_true, y_pred, pos_label="abnormal",
                                    zero_division=0)
        rec_ab    = recall_score(y_true, y_pred, pos_label="abnormal",
                                 zero_division=0)
        f1_ab     = f1_score(y_true, y_pred, pos_label="abnormal",
                             zero_division=0)
        spec_norm = float("nan")
        f1_mac    = float("nan")

    return {
        "n_records":          n,
        "n_normal":           n_normal,
        "n_abnormal":         n_abnormal,
        "pct_normal":         round(n_normal / n, 4),
        "pct_abnormal":       round(n_abnormal / n, 4),
        "accuracy":           round(acc, 4),
        "balanced_accuracy":  round(bal_acc, 4) if not np.isnan(bal_acc) else float("nan"),
        "precision_abnormal": round(prec_ab, 4)  if not np.isnan(prec_ab)  else float("nan"),
        "recall_abnormal":    round(rec_ab, 4)   if not np.isnan(rec_ab)   else float("nan"),
        "f1_abnormal":        round(f1_ab, 4)    if not np.isnan(f1_ab)    else float("nan"),
        "specificity_normal": round(spec_norm, 4) if not np.isnan(spec_norm) else float("nan"),
        "f1_macro":           round(f1_mac, 4)   if not np.isnan(f1_mac)   else float("nan"),
        "has_both_classes":   has_both,
        "dominant_class":     dominant,
        "n_errors":           n_err,
    }


def _classify_case(row: pd.Series) -> list[str]:
    """Return list of category tags that apply to a case row."""
    cats: list[str] = []
    pct_n  = row["pct_normal"]
    pct_ab = row["pct_abnormal"]
    acc    = row["accuracy"]
    rec_ab = row["recall_abnormal"] if not pd.isna(row["recall_abnormal"]) else 0.0
    n_ab   = row["n_abnormal"]
    both   = row["has_both_classes"]

    if pct_n >= 0.90 and acc >= 0.85:
        cats.append("normal_good")
    if pct_ab >= 0.90 and rec_ab >= 0.85 and acc >= 0.80:
        cats.append("abnormal_good")
    if both and 0.25 <= pct_n <= 0.75 and 0.25 <= pct_ab <= 0.75 and acc >= 0.75:
        cats.append("mixed_good")
    if both and acc < 0.65:
        cats.append("mixed_hard")
    if rec_ab >= 0.90 and n_ab >= 100:
        cats.append("abnormal_recall_good")
    return cats


# ---------------------------------------------------------------------------
# Auto-select at most 5 demo cases
# ---------------------------------------------------------------------------
_CATEGORY_PRIORITY = [
    "normal_good",
    "abnormal_good",
    "mixed_good",
    "abnormal_recall_good",
    "mixed_hard",           # optional — shows model limitations
]

_CATEGORY_META = {
    "normal_good":         ("Principalmente normal",    "normal",   "normal"),
    "abnormal_good":       ("Principalmente anormal",   "abnormal", "abnormal"),
    "mixed_good":          ("Caso mixto (bien clasificado)", "mixed",  "mixed"),
    "abnormal_recall_good":("Alta detección anormal",   "abnormal", "abnormal"),
    "mixed_hard":          ("Caso difícil (limitación)", "difficult","difficult"),
}


def _pick_best(df: pd.DataFrame, category: str) -> pd.Series | None:
    """Return best row for a given category (highest accuracy, then f1_macro)."""
    mask = df["categories"].apply(lambda cats: category in cats)
    sub  = df[mask].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(
        ["accuracy", "f1_macro"],
        ascending=False,
        na_position="last",
    )
    return sub.iloc[0]


def _build_demo_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Select up to 5 cases and build demo_cases_binary.csv rows."""
    rows     = []
    seen_ids: set[int] = set()

    for cat in _CATEGORY_PRIORITY:
        row = _pick_best(df, cat)
        if row is None:
            continue
        cid = int(row["case_id"])
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        title, btype, etype = _CATEGORY_META[cat]
        n_abn   = int(row["n_abnormal"])
        n_nor   = int(row["n_normal"])
        n_total = int(row["n_records"])
        acc     = row["accuracy"]
        rec_ab  = row["recall_abnormal"]
        pct_n   = row["pct_normal"]
        pct_ab  = row["pct_abnormal"]

        description = (
            f"Caso con {n_total} registros: "
            f"{n_nor} normales ({pct_n:.0%}) y "
            f"{n_abn} anormales ({pct_ab:.0%}). "
            f"Accuracy={acc:.3f}."
        )
        expected = (
            "Mayoría normales" if pct_n >= 0.90 else
            "Mayoría anormales" if pct_ab >= 0.90 else
            "Mezcla normal/anormal"
        )
        notes = (
            f"Seleccionado como {cat}. "
            f"recall_abnormal={rec_ab:.3f}" if not pd.isna(rec_ab) else
            f"Seleccionado como {cat}."
        )

        rows.append({
            "case_id":             cid,
            "title":               title,
            "binary_type":         btype,
            "description":         description,
            "expected_pattern":    expected,
            "n_beats":             n_total,
            "accuracy":            round(float(acc), 4),
            "balanced_accuracy":   round(float(row["balanced_accuracy"]), 4)
                                   if not pd.isna(row["balanced_accuracy"]) else "",
            "precision_abnormal":  round(float(row["precision_abnormal"]), 4)
                                   if not pd.isna(row["precision_abnormal"]) else "",
            "recall_abnormal":     round(float(rec_ab), 4)
                                   if not pd.isna(rec_ab) else "",
            "f1_abnormal":         round(float(row["f1_abnormal"]), 4)
                                   if not pd.isna(row["f1_abnormal"]) else "",
            "specificity_normal":  round(float(row["specificity_normal"]), 4)
                                   if not pd.isna(row["specificity_normal"]) else "",
            "notes":               notes,
        })

        if len(rows) >= 5:
            break

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 65)
    print("  05_select_binary_demo_cases.py")
    print("=" * 65)

    # -- Validate inputs -------------------------------------------------------
    for p in (PARQUET_PATH, MODEL_PATH, META_PATH):
        if not p.exists():
            print(f"\n  [ERROR] No encontrado: {p.relative_to(PROJECT_ROOT)}")
            sys.exit(1)

    # -- Load ------------------------------------------------------------------
    print("\n[1/6] Cargando artefactos…")
    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)

    numeric_features     = meta["numeric_features"]
    categorical_features = meta["categorical_features"]
    feature_cols         = numeric_features + categorical_features

    print(f"  features: {len(numeric_features)} num + {len(categorical_features)} cat "
          f"= {len(feature_cols)} total")

    print("  cargando parquet…")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  parquet: {df.shape[0]:,} filas · {df.shape[1]} columnas · "
          f"{df['case_id'].nunique()} casos")

    print("  cargando modelo…")
    pipeline = joblib.load(MODEL_PATH)

    # -- Build binary target ---------------------------------------------------
    print("\n[2/6] Construyendo target binario…")
    df["y_true"] = np.where(df["rhythm_label"] == "N", "normal", "abnormal")
    print(f"  normal   : {(df['y_true']=='normal').sum():,}")
    print(f"  abnormal : {(df['y_true']=='abnormal').sum():,}")

    # -- Batch predict (all cases at once) -------------------------------------
    print("\n[3/6] Prediciendo sobre todo el dataset (batch)…")
    X = df[feature_cols]
    raw_pred = pipeline.predict(X)
    df["y_pred"] = _norm_pred(raw_pred)
    print(f"  pred normal   : {(df['y_pred']=='normal').sum():,}")
    print(f"  pred abnormal : {(df['y_pred']=='abnormal').sum():,}")
    print(f"  accuracy global: {accuracy_score(df['y_true'], df['y_pred']):.4f}")

    # -- Per-case metrics ------------------------------------------------------
    print("\n[4/6] Calculando métricas por caso…")
    records = []
    case_ids = sorted(df["case_id"].unique())

    for cid in case_ids:
        mask   = df["case_id"] == cid
        yt     = df.loc[mask, "y_true"].values
        yp     = df.loc[mask, "y_pred"].values
        m      = _case_metrics(yt, yp)
        m["case_id"]    = cid
        m["categories"] = _classify_case(pd.Series(m))
        records.append(m)

    df_metrics = pd.DataFrame(records)
    df_metrics = df_metrics.sort_values("accuracy", ascending=False).reset_index(drop=True)

    # Summary
    n_both  = df_metrics["has_both_classes"].sum()
    n_only_n = (df_metrics["n_abnormal"] == 0).sum()
    n_only_a = (df_metrics["n_normal"]   == 0).sum()
    print(f"  {len(df_metrics)} casos evaluados")
    print(f"  {n_both} tienen ambas clases | {n_only_n} solo normal | {n_only_a} solo anormal")

    # Category counts
    cat_counts: dict[str, int] = {}
    for cats in df_metrics["categories"]:
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1
    print("  Categorías encontradas:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:30s}: {cnt}")

    # -- Save full metrics CSV -------------------------------------------------
    print("\n[5/6] Guardando CSVs…")
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_NPY_DIR.mkdir(parents=True, exist_ok=True)

    # Save readable version (without list column)
    df_save = df_metrics.drop(columns=["categories"]).copy()
    df_save.to_csv(OUT_METRICS, index=False)
    print(f"  [ok] {OUT_METRICS.relative_to(PROJECT_ROOT)}")

    # Candidates: only cases that have at least one category tag
    df_with_cats = df_metrics[df_metrics["categories"].apply(len) > 0].copy()
    df_with_cats = df_with_cats.assign(
        categories_str=df_with_cats["categories"].apply(lambda x: "|".join(x))
    ).drop(columns=["categories"])
    df_with_cats.to_csv(OUT_CANDIDATES, index=False)
    print(f"  [ok] {OUT_CANDIDATES.relative_to(PROJECT_ROOT)}")

    # -- Build and save demo catalogue -----------------------------------------
    df_demo = _build_demo_catalogue(df_metrics)
    df_demo.to_csv(OUT_DEMO_CSV, index=False)
    print(f"  [ok] {OUT_DEMO_CSV.relative_to(PROJECT_ROOT)}")

    # -- Copy .npy files -------------------------------------------------------
    print("\n[6/6] Copiando archivos .npy demo…")
    demo_case_ids = df_demo["case_id"].tolist()
    npy_copied = 0
    npy_missing = []

    for cid in demo_case_ids:
        src = DEMO_NPY_DIR / f"case_{cid}.npy"
        dst = ARTIFACTS_NPY_DIR / f"case_{cid}.npy"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  [ok]   case_{cid}.npy  →  {dst.relative_to(PROJECT_ROOT)}")
            npy_copied += 1
        else:
            print(f"  [warn] case_{cid}.npy no encontrado en {DEMO_NPY_DIR.relative_to(PROJECT_ROOT)}")
            npy_missing.append(cid)

    # -- Console report --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  TOP 10 CASOS POR ACCURACY")
    print("=" * 65)
    top_acc = df_metrics.nlargest(10, "accuracy")[
        ["case_id", "n_records", "n_normal", "n_abnormal",
         "accuracy", "balanced_accuracy", "recall_abnormal", "f1_macro"]
    ]
    print(top_acc.to_string(index=False))

    print("\n" + "=" * 65)
    print("  TOP 10 CASOS POR RECALL_ABNORMAL (requiere n_abnormal >= 50)")
    print("=" * 65)
    top_rec = (
        df_metrics[df_metrics["n_abnormal"] >= 50]
        .nlargest(10, "recall_abnormal")[
            ["case_id", "n_records", "n_abnormal",
             "recall_abnormal", "precision_abnormal", "f1_abnormal",
             "accuracy", "balanced_accuracy"]
        ]
    )
    print(top_rec.to_string(index=False))

    print("\n" + "=" * 65)
    print("  CASOS SELECCIONADOS PARA DEMO")
    print("=" * 65)
    if df_demo.empty:
        print("  (ningún caso seleccionado — revisar umbrales)")
    else:
        print(df_demo[["case_id", "binary_type", "accuracy",
                        "balanced_accuracy", "recall_abnormal",
                        "f1_abnormal", "n_beats"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    print(f"  Casos evaluados      : {len(df_metrics)}")
    print(f"  Casos con categoría  : {len(df_with_cats)}")
    print(f"  Casos demo elegidos  : {len(df_demo)}")
    print(f"  .npy copiados        : {npy_copied}/{len(demo_case_ids)}")
    if npy_missing:
        print(f"  .npy faltantes       : {npy_missing}")
    print()
    print(f"  {OUT_METRICS.relative_to(PROJECT_ROOT)}")
    print(f"  {OUT_CANDIDATES.relative_to(PROJECT_ROOT)}")
    print(f"  {OUT_DEMO_CSV.relative_to(PROJECT_ROOT)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
