"""06_prepare_selected_demo_npy.py

Extrae fragmentos demo de 60 s de los archivos .npy crudos de VitalDB
para los 4 casos seleccionados de la app Streamlit.

Ejecutar desde la raiz del proyecto:
    python scripts/06_prepare_selected_demo_npy.py

Busca cada caso en este orden:
  1. data/raw/vitaldb_waveforms/case_<id>.npy  (preferido: señal completa)
  2. data/demo/npy_cases/case_<id>.npy         (fragmento ya extraído)
  3. frontend/app/app_artifacts/demo/npy_cases/case_<id>.npy

Salidas:
  data/demo/npy_cases/case_<id>.npy
  frontend/app/app_artifacts/demo/npy_cases/case_<id>.npy
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEMO_CASES     = [5377, 2040, 337, 3098]
FS             = 500          # Hz
FRAGMENT_S     = 60           # seconds to keep
FRAGMENT_N     = FRAGMENT_S * FS  # 30 000 samples
MAX_NAN_PCT    = 0.02         # tolerate up to 2 % NaN in chosen fragment
SCAN_STEP      = FS * 30      # scan step when searching for clean window (30 s)

RAW_DIR        = PROJECT_ROOT / "data" / "raw" / "vitaldb_waveforms"
DEMO_DIR       = PROJECT_ROOT / "data" / "demo" / "npy_cases"
ARTIFACTS_DIR  = PROJECT_ROOT / "frontend" / "app" / "app_artifacts" / "demo" / "npy_cases"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ok(msg: str) -> None:   print(f"  [ok]   {msg}")
def _warn(msg: str) -> None: print(f"  [warn] {msg}")
def _info(msg: str) -> None: print(f"  [info] {msg}")


def _find_best_fragment(sig: np.ndarray, n: int, step: int, max_nan: float) -> np.ndarray | None:
    """Return the first n-sample window with NaN% <= max_nan.

    Scans with `step` stride.  Falls back to the window with minimum NaN
    if no clean window exists.
    """
    total = len(sig)
    if total < n:
        return None

    best_start = 0
    best_nan   = float("inf")

    for start in range(0, total - n + 1, step):
        window    = sig[start : start + n]
        nan_count = int(np.isnan(window).sum())
        nan_pct   = nan_count / n
        if nan_pct <= max_nan:
            return window.copy()
        if nan_pct < best_nan:
            best_nan   = nan_pct
            best_start = start

    # No clean window found — return best anyway, NaN will be ffill-padded
    window = sig[best_start : best_start + n].copy()
    _warn(f"No clean window found (best NaN={best_nan:.1%}). Using window at {best_start/FS:.0f}s.")
    return window


def _pad_nan(fragment: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values so the fragment is usable for display."""
    out = fragment.copy()
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out
    # Forward fill
    prev = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            if not np.isnan(prev):
                out[i] = prev
        else:
            prev = out[i]
    # Backward fill remaining leading NaN
    for i in range(len(out) - 1, -1, -1):
        if np.isnan(out[i]):
            j = i + 1
            while j < len(out) and np.isnan(out[j]):
                j += 1
            if j < len(out):
                out[i] = out[j]
    return out


def _process_case(case_id: int) -> dict:
    """Find, extract, save and return result info for one case."""
    fname = f"case_{case_id}.npy"
    result = {
        "case_id":   case_id,
        "status":    "missing",
        "source":    None,
        "demo_path": None,
        "art_path":  None,
        "size_kb":   None,
        "duration_s": None,
        "nan_pct":   None,
    }

    # --- Locate source -------------------------------------------------------
    candidates = [
        ("raw waveforms",  RAW_DIR / fname),
        ("demo npy_cases", DEMO_DIR / fname),
        ("app_artifacts",  ARTIFACTS_DIR / fname),
    ]
    src_label: str | None = None
    src_path:  Path | None = None

    for label, path in candidates:
        if path.exists():
            src_label = label
            src_path  = path
            break

    if src_path is None:
        _warn(f"case_{case_id}: no encontrado en ninguna ubicacion conocida.")
        return result

    result["source"] = src_label
    _info(f"case_{case_id}: encontrado en {src_label} ({src_path.stat().st_size/1024:.0f} KB)")

    # --- Load ----------------------------------------------------------------
    sig = np.load(str(src_path)).ravel().astype(np.float32)
    total_s = len(sig) / FS

    # --- Extract fragment ----------------------------------------------------
    if len(sig) <= FRAGMENT_N:
        # Already a short fragment — use as-is
        fragment = sig.copy()
        _info(f"case_{case_id}: señal ya es corta ({total_s:.0f}s), usando completa")
    else:
        fragment = _find_best_fragment(sig, FRAGMENT_N, SCAN_STEP, MAX_NAN_PCT)
        if fragment is None:
            _warn(f"case_{case_id}: señal demasiado corta ({total_s:.0f}s < {FRAGMENT_S}s)")
            return result

    # Pad NaN to avoid display artifacts
    nan_before = float(np.isnan(fragment).mean())
    fragment   = _pad_nan(fragment)
    nan_after  = float(np.isnan(fragment).mean())
    if nan_before > 0:
        _info(f"case_{case_id}: NaN padded ({nan_before:.1%} -> {nan_after:.1%})")

    # --- Save ----------------------------------------------------------------
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    demo_dst = DEMO_DIR / fname
    art_dst  = ARTIFACTS_DIR / fname

    np.save(str(demo_dst), fragment)
    shutil.copy2(demo_dst, art_dst)

    size_kb = demo_dst.stat().st_size / 1024
    _ok(f"case_{case_id}: guardado {len(fragment)/FS:.0f}s  ({size_kb:.1f} KB)  "
        f"-> {demo_dst.relative_to(PROJECT_ROOT)}")
    _ok(f"case_{case_id}: copiado  -> {art_dst.relative_to(PROJECT_ROOT)}")

    result.update({
        "status":    "ok",
        "demo_path": str(demo_dst.relative_to(PROJECT_ROOT)),
        "art_path":  str(art_dst.relative_to(PROJECT_ROOT)),
        "size_kb":   round(size_kb, 1),
        "duration_s": round(len(fragment) / FS, 1),
        "nan_pct":   round(nan_after * 100, 2),
    })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 62)
    print("  06_prepare_selected_demo_npy.py")
    print("=" * 62)
    print(f"\n  Casos: {DEMO_CASES}")
    print(f"  Fragmento: {FRAGMENT_S}s  ({FRAGMENT_N} muestras a {FS} Hz)")
    print(f"  Destinos:")
    print(f"    {DEMO_DIR.relative_to(PROJECT_ROOT)}")
    print(f"    {ARTIFACTS_DIR.relative_to(PROJECT_ROOT)}")

    results = []
    for cid in DEMO_CASES:
        print(f"\n[case_{cid}]")
        r = _process_case(cid)
        results.append(r)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    ok_cases   = [r for r in results if r["status"] == "ok"]
    miss_cases = [r for r in results if r["status"] == "missing"]

    print("\n" + "=" * 62)
    print("  RESUMEN FINAL")
    print("=" * 62)
    print(f"\n  Casos procesados : {len(ok_cases)}/{len(DEMO_CASES)}")

    if ok_cases:
        print("\n  Archivos generados:")
        for r in ok_cases:
            print(f"    case_{r['case_id']}: {r['duration_s']}s  {r['size_kb']} KB  "
                  f"NaN={r['nan_pct']}%  (fuente: {r['source']})")

    if miss_cases:
        print(f"\n  [MISSING] Casos no encontrados ({len(miss_cases)}):")
        for r in miss_cases:
            print(f"    case_{r['case_id']}")
        print("\n  Para obtener los .npy crudos, descargarlos manualmente de VitalDB")
        print("  y guardarlos en data/raw/vitaldb_waveforms/")

    print("\n  Rutas en app_artifacts (para subir a GitHub):")
    for cid in DEMO_CASES:
        dst = ARTIFACTS_DIR / f"case_{cid}.npy"
        status = "[ok]  " if dst.exists() else "[miss]"
        size   = f"{dst.stat().st_size/1024:.1f} KB" if dst.exists() else "—"
        print(f"    {status} {dst.relative_to(PROJECT_ROOT)}  {size}")

    print("\n  Git — verificar que no esten ignorados:")
    for cid in DEMO_CASES:
        print(f"    git check-ignore -v frontend/app/app_artifacts/demo/npy_cases/case_{cid}.npy")

    print("=" * 62 + "\n")

    if miss_cases:
        sys.exit(1)


if __name__ == "__main__":
    main()
