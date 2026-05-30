from pathlib import Path

# Anchored to THIS file: frontend/app/utils/paths.py
_UTILS_DIR   = Path(__file__).resolve().parent   # frontend/app/utils/
APP_DIR      = _UTILS_DIR.parent                  # frontend/app/
FRONTEND_DIR = APP_DIR.parent                     # frontend/
PROJECT_ROOT = FRONTEND_DIR.parent                # project root

REPORTS_DIR        = PROJECT_ROOT / "reports"
REPORT_TABLES_DIR  = REPORTS_DIR  / "tables"
REPORT_FIGURES_DIR = REPORTS_DIR  / "figures"
MODELS_DIR         = PROJECT_ROOT / "models"
DATA_DIR           = PROJECT_ROOT / "data"
DEMO_DATA_DIR      = DATA_DIR     / "demo"
ASSETS_DIR         = APP_DIR      / "assets"

# ---------------------------------------------------------------------------
# App artifacts — small, git-tracked files for Streamlit Cloud deployment.
# Populated by scripts/04_prepare_streamlit_artifacts.py.
# Loaders check these paths FIRST, then fall back to the project-root paths above.
# ---------------------------------------------------------------------------
APP_ARTIFACTS_DIR    = APP_DIR / "app_artifacts"
ARTIFACTS_MODELS_DIR = APP_ARTIFACTS_DIR / "models"
ARTIFACTS_TABLES_DIR = APP_ARTIFACTS_DIR / "reports" / "tables"
ARTIFACTS_FIGURES_DIR = APP_ARTIFACTS_DIR / "reports" / "figures"
ARTIFACTS_DEMO_DIR   = APP_ARTIFACTS_DIR / "demo"
ARTIFACTS_NPY_DIR    = ARTIFACTS_DEMO_DIR / "npy_cases"
DEMO_CASES_CSV       = ARTIFACTS_DEMO_DIR / "demo_cases_binary.csv"


def resolve_path(*candidates: Path) -> "Path | None":
    """Return the first existing path from the candidates list, or None."""
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


def resolve_npy_dir() -> Path:
    """Return the npy_cases directory, preferring app_artifacts over data/demo."""
    if ARTIFACTS_NPY_DIR.exists():
        return ARTIFACTS_NPY_DIR
    return DEMO_DATA_DIR / "npy_cases"
