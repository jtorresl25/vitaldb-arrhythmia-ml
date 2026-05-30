"""04_prepare_streamlit_artifacts.py

Prepara la carpeta frontend/app/app_artifacts/ con los artefactos necesarios
para que Streamlit Cloud funcione sin depender de archivos locales pesados.

Ejecutar desde la raiz del proyecto:
    python scripts/04_prepare_streamlit_artifacts.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODELS_DIR         = PROJECT_ROOT / "models"
REPORT_TABLES_DIR  = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DEMO_NPY_DIR       = PROJECT_ROOT / "data" / "demo" / "npy_cases"
RAW_WAVEFORMS_DIR  = PROJECT_ROOT / "data" / "raw" / "vitaldb_waveforms"

APP_ARTIFACTS_DIR    = PROJECT_ROOT / "frontend" / "app" / "app_artifacts"
ARTIFACTS_MODELS_DIR = APP_ARTIFACTS_DIR / "models"
ARTIFACTS_TABLES_DIR = APP_ARTIFACTS_DIR / "reports" / "tables"
ARTIFACTS_FIGURES_DIR = APP_ARTIFACTS_DIR / "reports" / "figures"
ARTIFACTS_DEMO_DIR   = APP_ARTIFACTS_DIR / "demo"
ARTIFACTS_NPY_DIR    = ARTIFACTS_DEMO_DIR / "npy_cases"

# Primary demo case for the downloadable ECG fragment
_PRIMARY_DEMO_CASE = 337
_FRAGMENT_SECONDS  = 60     # max seconds to extract from a raw waveform
_FRAGMENT_FS       = 500    # assumed sample rate

# ---------------------------------------------------------------------------
# Required model files
# ---------------------------------------------------------------------------
_REQUIRED_FILES: list[tuple[Path, Path]] = [
    (MODELS_DIR / "tabular_best_model_pipeline.joblib",
     ARTIFACTS_MODELS_DIR / "tabular_best_model_pipeline.joblib"),
    (MODELS_DIR / "tabular_best_model_metadata.json",
     ARTIFACTS_MODELS_DIR / "tabular_best_model_metadata.json"),
]

# ---------------------------------------------------------------------------
# Optional CSV tables
# ---------------------------------------------------------------------------
_OPTIONAL_TABLES: list[tuple[str, str]] = [
    # Modelo final oficial y benchmark exploratorio
    ("tabular_model_final_official.csv",            "tabular_model_final_official.csv"),
    ("tabular_model_comparison_history.csv",        "tabular_model_comparison_history.csv"),
    # Corrida final (1 fila)
    ("tabular_model_comparison_test.csv",           "tabular_model_comparison_test.csv"),
    ("tabular_model_comparison_cv.csv",             "tabular_model_comparison_cv.csv"),
    # Reportes del modelo ganador
    ("tabular_best_model_classification_report.csv","tabular_best_model_classification_report.csv"),
    ("tabular_binary_metrics.csv",                  "tabular_binary_metrics.csv"),
    ("tabular_confusion_matrix_absolute.csv",       "tabular_confusion_matrix_absolute.csv"),
    ("confusion_matrix.csv",                        "confusion_matrix.csv"),
    ("tabular_feature_importance_best_model.csv",   "tabular_feature_importance_best_model.csv"),
    ("tabular_feature_list_used.csv",               "tabular_feature_list_used.csv"),
    ("tabular_train_test_split_summary.csv",        "tabular_train_test_split_summary.csv"),
    ("tabular_class_support_train_test.csv",        "tabular_class_support_train_test.csv"),
    ("binary_case_level_metrics.csv",               "binary_case_level_metrics.csv"),
    ("binary_demo_case_candidates.csv",             "binary_demo_case_candidates.csv"),
]

# ---------------------------------------------------------------------------
# Optional figure files
# ---------------------------------------------------------------------------
_OPTIONAL_FIGURES: list[tuple[str, str]] = [
    ("tabular_best_model_confusion_matrix_absolute.png",
     "tabular_best_model_confusion_matrix_absolute.png"),
    ("tabular_best_model_confusion_matrix_normalized.png",
     "tabular_best_model_confusion_matrix_normalized.png"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")

def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")

def _err(msg: str) -> None:
    print(f"  [ERR]  {msg}")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"  [dir]  {path.relative_to(PROJECT_ROOT)}")


def _copy(src: Path, dst: Path, optional: bool = False) -> bool:
    if not src.exists():
        if optional:
            _warn(f"No encontrado: {src.relative_to(PROJECT_ROOT)}")
        else:
            _err(f"FALTANTE (obligatorio): {src.relative_to(PROJECT_ROOT)}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size_kb = src.stat().st_size / 1024
    src_rel = src.relative_to(PROJECT_ROOT)
    dst_rel = dst.relative_to(PROJECT_ROOT)
    _ok(f"{src_rel} -> {dst_rel}  ({size_kb:.1f} KB)")
    return True


def _extract_npy_fragment(src: Path, dst: Path, seconds: int, fs: int) -> bool:
    """Extract first `seconds` seconds from a .npy waveform and save to dst."""
    try:
        import numpy as np
        sig = np.load(str(src))
        sig = sig.ravel()
        n = min(len(sig), seconds * fs)
        if n < fs * 5:
            _warn(f"Fragmento demasiado corto ({n} muestras) en {src.name}")
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(dst), sig[:n])
        size_kb = dst.stat().st_size / 1024
        _ok(f"{src.relative_to(PROJECT_ROOT)} [{n} muestras] -> {dst.relative_to(PROJECT_ROOT)}  ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        _warn(f"Error extrayendo fragmento de {src.name}: {e}")
        return False


def _ensure_case_npy(case_id: int) -> bool:
    """
    Find or extract case_<id>.npy for the demo.
    Search order:
      1. demo npy_cases (already extracted fragment)
      2. app_artifacts npy_cases (already there)
      3. raw vitaldb_waveforms (extract fragment)
    Returns True if the file ends up in ARTIFACTS_NPY_DIR.
    """
    fname = f"case_{case_id}.npy"
    dst   = ARTIFACTS_NPY_DIR / fname

    # Already in app_artifacts?
    if dst.exists():
        _ok(f"{fname} ya existe en app_artifacts")
        return True

    # Already extracted in demo npy_cases?
    demo_src = DEMO_NPY_DIR / fname
    if demo_src.exists():
        return _copy(demo_src, dst, optional=False)

    # Raw waveforms?
    raw_src = RAW_WAVEFORMS_DIR / fname
    if raw_src.exists():
        print(f"  [info] Extrayendo fragmento de {_FRAGMENT_SECONDS}s de {raw_src.name}...")
        ok = _extract_npy_fragment(raw_src, DEMO_NPY_DIR / fname, _FRAGMENT_SECONDS, _FRAGMENT_FS)
        if ok:
            return _copy(DEMO_NPY_DIR / fname, dst, optional=False)

    _warn(f"{fname} no encontrado en data/demo/npy_cases/, app_artifacts/ ni data/raw/vitaldb_waveforms/")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 62)
    print("  04_prepare_streamlit_artifacts.py")
    print("=" * 62)

    # 1. Folder structure
    print("\n[1/6] Creando estructura de carpetas...")
    for d in (ARTIFACTS_MODELS_DIR, ARTIFACTS_TABLES_DIR,
              ARTIFACTS_FIGURES_DIR, ARTIFACTS_NPY_DIR):
        _mkdir(d)

    # 2. Required model files
    print("\n[2/6] Copiando archivos de modelo (obligatorios)...")
    missing_required: list[str] = []
    for src, dst in _REQUIRED_FILES:
        if not _copy(src, dst, optional=False):
            missing_required.append(str(src.relative_to(PROJECT_ROOT)))

    # 3. Optional CSV tables
    print("\n[3/6] Copiando tablas de reportes...")
    for src_name, dst_name in _OPTIONAL_TABLES:
        _copy(REPORT_TABLES_DIR / src_name,
              ARTIFACTS_TABLES_DIR / dst_name,
              optional=True)

    # 4. Optional figures
    print("\n[4/6] Copiando figuras...")
    for src_name, dst_name in _OPTIONAL_FIGURES:
        _copy(REPORT_FIGURES_DIR / src_name,
              ARTIFACTS_FIGURES_DIR / dst_name,
              optional=True)

    # 5. Primary demo case .npy (case_337 — mixto representativo)
    print(f"\n[5/6] Caso demo principal: case_{_PRIMARY_DEMO_CASE}.npy...")
    primary_ok = _ensure_case_npy(_PRIMARY_DEMO_CASE)

    # 6. Copy demo_cases_binary.csv (already managed by scripts/05_select_binary_demo_cases.py)
    print("\n[6/6] Copiando demo_cases_binary.csv...")
    demo_csv_src = ARTIFACTS_DEMO_DIR / "demo_cases_binary.csv"
    if demo_csv_src.exists():
        _ok(f"demo_cases_binary.csv ya en {ARTIFACTS_DEMO_DIR.relative_to(PROJECT_ROOT)}")
    else:
        _warn("demo_cases_binary.csv no encontrado. Ejecuta scripts/05_select_binary_demo_cases.py primero.")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    all_files = sorted(f for f in APP_ARTIFACTS_DIR.rglob("*") if f.is_file())
    total_size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)

    print("\n" + "=" * 62)
    print("  RESUMEN FINAL")
    print("=" * 62)

    if missing_required:
        print(f"\n  [ERR] Archivos OBLIGATORIOS faltantes ({len(missing_required)}):")
        for f in missing_required:
            print(f"        - {f}")
        print("\n  La app no podra cargar el modelo en Streamlit Cloud.")
    else:
        print("\n  [ok]  Todos los archivos obligatorios copiados.")

    print(f"\n  case_{_PRIMARY_DEMO_CASE}.npy: {'disponible' if primary_ok else 'NO DISPONIBLE'}")
    print(f"\n  Carpeta: {APP_ARTIFACTS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Archivos: {len(all_files)}")
    print(f"  Tamanio:  {total_size_mb:.1f} MB")

    print("\n  Archivos para subir a GitHub (Streamlit Cloud):")
    for f in all_files:
        rel = f.relative_to(PROJECT_ROOT)
        print(f"    {rel}")

    print("\n  Git — agregar con force si .gitignore interfiere:")
    print("    git add -f frontend/app/app_artifacts/")
    print("\n  Verificar que no esten ignorados:")
    print("    git check-ignore -v frontend/app/app_artifacts/models/tabular_best_model_pipeline.joblib")
    print("    git check-ignore -v frontend/app/app_artifacts/reports/tables/tabular_model_final_official.csv")
    print("    git check-ignore -v frontend/app/app_artifacts/demo/demo_cases_binary.csv")
    print("=" * 62 + "\n")

    if missing_required:
        sys.exit(1)


if __name__ == "__main__":
    main()
