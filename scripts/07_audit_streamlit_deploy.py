"""07_audit_streamlit_deploy.py

Auditoría de artefactos para deploy en Streamlit Cloud.
Verifica que todos los archivos obligatorios estén presentes, legibles y coherentes.

Ejecutar desde la raíz del proyecto:
    python scripts/07_audit_streamlit_deploy.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows so box/emoji characters print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
APP_ARTIFACTS = PROJECT_ROOT / "frontend" / "app" / "app_artifacts"

# ---------------------------------------------------------------------------
# Archivo → (obligatorio, descripción breve)
# ---------------------------------------------------------------------------
REQUIRED: list[tuple[Path, str]] = [
    (APP_ARTIFACTS / "models" / "tabular_best_model_pipeline.joblib",
     "Modelo LinearSVC (pipeline joblib)"),
    (APP_ARTIFACTS / "models" / "tabular_best_model_metadata.json",
     "Metadata del modelo (features, métricas, winner)"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_model_final_official.csv",
     "Registro oficial del modelo final"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_model_comparison_history.csv",
     "Histórico de benchmarks exploratorios"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_model_comparison_test.csv",
     "Comparativa de modelos — test"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_model_comparison_cv.csv",
     "Comparativa de modelos — CV"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_best_model_classification_report.csv",
     "Classification report (por clase)"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_binary_metrics.csv",
     "Métricas binarias globales"),
    (APP_ARTIFACTS / "reports" / "tables" / "confusion_matrix.csv",
     "Matriz de confusión (formato largo)"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_confusion_matrix_absolute.csv",
     "Matriz de confusión (formato ancho)"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_feature_importance_best_model.csv",
     "Importancia de features"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_feature_list_used.csv",
     "Lista de features usadas"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_train_test_split_summary.csv",
     "Resumen train/test split"),
    (APP_ARTIFACTS / "reports" / "tables" / "tabular_class_support_train_test.csv",
     "Soporte por clase en train/test"),
    (APP_ARTIFACTS / "reports" / "tables" / "binary_case_level_metrics.csv",
     "Métricas por case_id (todos los casos)"),
    (APP_ARTIFACTS / "reports" / "tables" / "binary_demo_case_candidates.csv",
     "Candidatos demo por categoría"),
    (APP_ARTIFACTS / "demo" / "demo_cases_binary.csv",
     "Catálogo de casos demo"),
    (APP_ARTIFACTS / "demo" / "case_features" / "case_337.parquet",
     "Features tabulares case_337 (mixto)"),
    (APP_ARTIFACTS / "demo" / "case_features" / "case_5377.parquet",
     "Features tabulares case_5377 (normal)"),
    (APP_ARTIFACTS / "demo" / "case_features" / "case_2040.parquet",
     "Features tabulares case_2040 (anormal)"),
    (APP_ARTIFACTS / "demo" / "case_features" / "case_1996.parquet",
     "Features tabulares case_1996 (mixto adicional)"),
]

OPTIONAL: list[tuple[Path, str]] = [
    (APP_ARTIFACTS / "demo" / "npy_cases" / "case_337.npy",
     "ECG .npy case_337"),
    (APP_ARTIFACTS / "demo" / "npy_cases" / "case_5377.npy",
     "ECG .npy case_5377"),
    (APP_ARTIFACTS / "demo" / "npy_cases" / "case_2040.npy",
     "ECG .npy case_2040"),
    (APP_ARTIFACTS / "demo" / "npy_cases" / "case_1996.npy",
     "ECG .npy case_1996"),
    (APP_ARTIFACTS / "reports" / "figures" / "tabular_best_model_confusion_matrix_absolute.png",
     "Figura CM absoluta"),
    (APP_ARTIFACTS / "reports" / "figures" / "tabular_best_model_confusion_matrix_normalized.png",
     "Figura CM normalizada"),
]

# Files that must NOT be present in the deploy path (too heavy)
FORBIDDEN_HEAVY: list[tuple[Path, str]] = [
    (PROJECT_ROOT / "data" / "processed" / "filtered_tabular_modeling_dataset.parquet",
     "Dataset tabular completo (~600 MB) — solo en local"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _size_str(path: Path) -> str:
    s = path.stat().st_size
    if s < 1024:
        return f"{s} B"
    if s < 1024 ** 2:
        return f"{s/1024:.1f} KB"
    return f"{s/1024**2:.1f} MB"


def _check_readable(path: Path) -> tuple[bool, str]:
    """Try to read the file based on its extension. Returns (ok, detail)."""
    ext = path.suffix.lower()
    try:
        if ext == ".json":
            with open(path, encoding="utf-8") as fh:
                json.load(fh)
            return True, "JSON válido"
        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(path, nrows=5)
            return True, f"{len(df.columns)} columnas"
        if ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            return True, f"{len(df):,} filas × {len(df.columns)} cols"
        if ext == ".joblib":
            import joblib
            obj = joblib.load(path)
            return True, f"{type(obj).__name__}"
        if ext == ".png":
            return True, "PNG presente"
        if ext == ".npy":
            import numpy as np
            arr = np.load(path, mmap_mode="r")
            return True, f"shape={arr.shape} dtype={arr.dtype}"
        return True, "existe"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def main() -> int:
    errors   = 0
    warnings = 0

    print()
    print("=" * 70)
    print("  07_audit_streamlit_deploy.py - Auditoria de artefactos de deploy")
    print("=" * 70)
    print(f"  PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"  APP_ARTIFACTS : {APP_ARTIFACTS}")
    print()

    # ── Obligatorios ──────────────────────────────────────────────────────────
    print("-" * 70)
    print("  ARCHIVOS OBLIGATORIOS")
    print("-" * 70)
    for path, desc in REQUIRED:
        rel = path.relative_to(PROJECT_ROOT)
        if not path.exists():
            print(f"  [ERROR] FALTANTE  {rel}")
            print(f"              {desc}")
            errors += 1
        else:
            ok, detail = _check_readable(path)
            size = _size_str(path)
            if ok:
                print(f"  [OK] OK        {rel}  [{size}]  {detail}")
            else:
                print(f"  [ERROR] ERROR     {rel}  [{size}]  no se puede leer: {detail}")
                errors += 1

    # ── Opcionales ────────────────────────────────────────────────────────────
    print()
    print("-" * 70)
    print("  ARCHIVOS OPCIONALES")
    print("-" * 70)
    for path, desc in OPTIONAL:
        rel = path.relative_to(PROJECT_ROOT)
        if not path.exists():
            print(f"  [WARN] AUSENTE   {rel}")
            print(f"              {desc} (no bloquea el deploy)")
            warnings += 1
        else:
            ok, detail = _check_readable(path)
            size = _size_str(path)
            if ok:
                print(f"  [OK] OK        {rel}  [{size}]")
            else:
                print(f"  [WARN] WARNING   {rel}  [{size}]  {detail}")
                warnings += 1

    # ── Archivos pesados que NO deben importar al deploy ──────────────────────
    print()
    print("-" * 70)
    print("  ARCHIVOS PESADOS (deben estar ausentes del deploy)")
    print("-" * 70)
    for path, desc in FORBIDDEN_HEAVY:
        rel = path.relative_to(PROJECT_ROOT)
        if path.exists():
            size = _size_str(path)
            print(f"  [INFO] LOCAL     {rel}  [{size}]  (solo local, no va a Streamlit Cloud)")
        else:
            print(f"  [OK] AUSENTE   {rel}  (OK para deploy)")

    # ── Validación cruzada: casos en demo_cases_binary.csv vs case_features ───
    print()
    print("-" * 70)
    print("  COHERENCIA: demo_cases_binary.csv ↔ case_features/")
    print("-" * 70)
    demo_csv = APP_ARTIFACTS / "demo" / "demo_cases_binary.csv"
    if demo_csv.exists():
        try:
            import pandas as pd
            df_demo = pd.read_csv(demo_csv)
            for _, row in df_demo.iterrows():
                cid = int(row["case_id"])
                feat_path = APP_ARTIFACTS / "demo" / "case_features" / f"case_{cid}.parquet"
                npy_path  = APP_ARTIFACTS / "demo" / "npy_cases"    / f"case_{cid}.npy"
                feat_ok = feat_path.exists()
                npy_ok  = npy_path.exists()
                feat_str = f"[OK] features" if feat_ok else "[ERROR] FALTANTE features"
                npy_str  = f"[OK] npy"      if npy_ok  else "[WARN] sin npy"
                if not feat_ok:
                    errors += 1
                if not npy_ok:
                    warnings += 1
                print(f"  case_{cid:5d}  {feat_str}  {npy_str}  [{row.get('binary_type','?')}] {row.get('title','')}")
        except Exception as exc:
            print(f"  [ERROR] No se pudo leer demo_cases_binary.csv: {exc}")
            errors += 1
    else:
        print("  [ERROR] demo_cases_binary.csv no encontrado")
        errors += 1

    # ── Metadata coherence ────────────────────────────────────────────────────
    print()
    print("-" * 70)
    print("  COHERENCIA: metadata JSON")
    print("-" * 70)
    meta_path = APP_ARTIFACTS / "models" / "tabular_best_model_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        n_num  = len(meta.get("numeric_features", []))
        n_cat  = len(meta.get("categorical_features", []))
        winner = meta.get("winner_model", "—")
        acc    = meta.get("test_accuracy", meta.get("accuracy", "—"))
        f1     = meta.get("test_f1_macro", meta.get("f1_macro", "—"))
        print(f"  winner_model      : {winner}")
        print(f"  numeric_features  : {n_num}")
        print(f"  categorical_feats : {n_cat}")
        print(f"  test_accuracy     : {acc}")
        print(f"  test_f1_macro     : {f1}")
        if n_num + n_cat == 0:
            print("  [WARN] Sin features en metadata")
            warnings += 1
        else:
            print(f"  [OK] {n_num + n_cat} features definidas")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    if errors == 0 and warnings == 0:
        print("  [OK] Todo OK — listo para Streamlit Cloud")
    elif errors == 0:
        print(f"  [OK] Sin errores críticos · {warnings} advertencia(s) menores")
    else:
        print(f"  [ERROR] {errors} error(es) crítico(s) · {warnings} advertencia(s)")
        print()
        print("  Comandos sugeridos para agregar archivos faltantes a git:")
        print("    git add -f frontend/app/app_artifacts/")
        print("    git status")
    print()
    print("  Comandos para deploy:")
    print("    python scripts/04_prepare_streamlit_artifacts.py   # regenerar artefactos")
    print("    python scripts/07_audit_streamlit_deploy.py        # re-auditar")
    print("    git add -f frontend/app/app_artifacts/")
    print("    git add frontend/app/ scripts/ .gitignore")
    print("    git status")
    print('    git commit -m "Finalize Streamlit binary arrhythmia app deployment"')
    print("    git push")
    print("=" * 70)
    print()

    return errors


if __name__ == "__main__":
    sys.exit(main())
