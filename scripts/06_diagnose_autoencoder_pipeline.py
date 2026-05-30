# -*- coding: utf-8 -*-
"""
scripts/06_diagnose_autoencoder_pipeline.py

Auditoria critica del autoencoder ECG en tres partes:

  Parte 1 - Diagnostico sobre resultados existentes (sin re-entrenamiento)
             Lee reports/autoencoder/autoencoder_test_results.csv y calcula
             estadisticas por grupo (Normal/Anomalo) y por clase.

  Parte 2 - Ablation study de preprocesamiento x tamano de ventana
             3 modos x 3 tamanos = 9 configs.
             Cada config entrena un mini-autoencoder (PyTorch, 10 epocas).

  Parte 3 - Recomendaciones automaticas

Uso rapido:
  .venv\\Scripts\\python.exe scripts/06_diagnose_autoencoder_pipeline.py
  .venv\\Scripts\\python.exe scripts/06_diagnose_autoencoder_pipeline.py --device cuda
  .venv\\Scripts\\python.exe scripts/06_diagnose_autoencoder_pipeline.py --skip-ablation

Salidas:
  reports/autoencoder/autoencoder_error_diagnostics.csv
  reports/autoencoder/autoencoder_ablation_results.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- project root -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNOTATION_FILENAME_REGEX,
    BEAT_TIME_COLUMN,
    DEFAULT_ECG_FS_HZ,
    DEFAULT_WINDOW_SECONDS,
    EXCLUDED_RHYTHM_LABELS,
    RANDOM_SEED,
    REPORTS_DIR,
    SIGNAL_QUALITY_COLUMN,
    TARGET_COLUMN,
    VITALDB_WAVEFORMS_DIR,
    PHYSIONET_DIR,
)
from src.pipeline import interpolate_nans, bandpass_filter, normalize_ecg
from src.windowing import build_windows_for_case

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print(
        "[ERROR] PyTorch no esta instalado.\n"
        "  .venv\\Scripts\\python.exe -m pip install torch"
    )
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
from scipy.signal import butter, filtfilt
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

# =============================================================================
# CONFIGURACION ABLATION (siempre modo rapido)
# =============================================================================

FS: int = DEFAULT_ECG_FS_HZ
NORMAL_LABEL: str = "N"
TEST_SIZE: float = 0.25

PREPROCESS_MODES: list[str] = [
    "current_pipeline",   # A: interp_nans + bandpass global + z-norm global + z-norm ventana
    "local_window_only",  # B: interp_nans + z-norm ventana (sin bandpass global)
    "local_bandpass",     # C: interp_nans + bandpass por ventana + z-norm ventana
]
WINDOW_SIZES_SEC: list[int] = [2, 4, 6]

ABLATION_MAX_CASES: int = 80
ABLATION_MAX_WINDOWS_PER_CASE: int = 100

ABLATION_HIDDEN_DIMS: tuple[int, ...] = (256, 64)
ABLATION_BOTTLENECK: int = 16
ABLATION_EPOCHS: int = 10
ABLATION_BATCH: int = 256
ABLATION_LR: float = 1e-3
ABLATION_PATIENCE: int = 3
ABLATION_VAL_SPLIT: float = 0.10

THRESHOLD_PERCENTILE: int = 95

DIAG_CSV = REPORTS_DIR / "autoencoder" / "autoencoder_error_diagnostics.csv"
ABLATION_CSV = REPORTS_DIR / "autoencoder" / "autoencoder_ablation_results.csv"
EXISTING_RESULTS_CSV = REPORTS_DIR / "autoencoder" / "autoencoder_test_results.csv"


# =============================================================================
# ARGPARSER
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostico del autoencoder ECG.")
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Dispositivo PyTorch (default: auto).",
    )
    p.add_argument(
        "--skip-ablation", action="store_true",
        help="Ejecutar solo Parte 1 (diagnostico), sin ablation.",
    )
    return p.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda solicitado pero CUDA no disponible.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# PARTE 1 - DIAGNOSTICO SOBRE RESULTADOS EXISTENTES
# =============================================================================

def run_diagnostics(log: logging.Logger) -> pd.DataFrame | None:
    """Lee el CSV existente y calcula estadisticas por grupo y por clase."""
    if not EXISTING_RESULTS_CSV.exists():
        log.warning("No se encontro %s — se omite Parte 1.", EXISTING_RESULTS_CSV)
        return None

    df = pd.read_csv(EXISTING_RESULTS_CSV)
    log.info("Diagnostico sobre %d ventanas de test.", len(df))

    scores = df["reconstruction_error"].values
    y = df["true_binary"].values
    labels = df["true_label"].values

    auc = roc_auc_score(y, scores)
    auc_inv = 1.0 - auc

    log.info("ROC-AUC (error)     = %.4f", auc)
    log.info("ROC-AUC (neg-error) = %.4f  (use if inverted)", auc_inv)

    rows: list[dict] = []

    def _stats(name: str, sub_scores: np.ndarray) -> dict:
        return {
            "group": name,
            "n_windows": len(sub_scores),
            "mean_error": round(float(sub_scores.mean()), 6),
            "median_error": round(float(np.median(sub_scores)), 6),
            "p50": round(float(np.percentile(sub_scores, 50)), 6),
            "p75": round(float(np.percentile(sub_scores, 75)), 6),
            "p90": round(float(np.percentile(sub_scores, 90)), 6),
            "p95": round(float(np.percentile(sub_scores, 95)), 6),
            "roc_auc_vs_normal": None,
        }

    mask_n = labels == NORMAL_LABEL
    rows.append(_stats("Normal", scores[mask_n]))
    rows.append(_stats("Anomalous_overall", scores[~mask_n]))

    normal_scores = scores[mask_n]
    for cls in sorted(df["true_label"].unique()):
        if cls == NORMAL_LABEL:
            continue
        mask_cls = labels == cls
        n_cls = int(mask_cls.sum())
        cls_scores = scores[mask_cls]
        d = _stats(f"Anomalous:{cls}", cls_scores)
        if n_cls >= 10:
            combined = np.concatenate([normal_scores, cls_scores])
            y_bin = np.concatenate([
                np.zeros(len(normal_scores), dtype=int),
                np.ones(n_cls, dtype=int),
            ])
            d["roc_auc_vs_normal"] = round(float(roc_auc_score(y_bin, combined)), 4)
        rows.append(d)

    diag_df = pd.DataFrame(rows)
    DIAG_CSV.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(DIAG_CSV, index=False)
    log.info("Diagnostico guardado -> %s", DIAG_CSV)

    print("\n-- DIAGNOSTICO DE ERRORES POR GRUPO/CLASE " + "-" * 37)
    print(f"{'Grupo':<35} {'N':>6} {'Mean':>8} {'Median':>8} {'p90':>8} {'AUC_vs_N':>10}")
    print("-" * 80)
    for _, r in diag_df.iterrows():
        auc_str = f"{r['roc_auc_vs_normal']:.4f}" if r["roc_auc_vs_normal"] is not None else "   n/a"
        print(
            f"{str(r['group']):<35} {int(r['n_windows']):>6} "
            f"{r['mean_error']:>8.4f} {r['median_error']:>8.4f} "
            f"{r['p90']:>8.4f} {auc_str:>10}"
        )
    print()
    print(f"  ROC-AUC global (error)     = {auc:.4f}")
    print(f"  ROC-AUC global (neg-error) = {auc_inv:.4f}")

    if auc < 0.5:
        print("  [!] Score INVERTIDO: anomalias reconstruyen MEJOR que normales.")
        print("      -> En produccion, usar  score = -reconstruction_error")
        print("      -> O invertir el threshold: flag si error < umbral_bajo")
    return diag_df


# =============================================================================
# PARTE 2 - ABLATION STUDY
# =============================================================================

# --- Carga de datos -----------------------------------------------------------

def _discover_annotation_files() -> dict[int, Path]:
    import re
    ann_dir = PHYSIONET_DIR / "Annotation_Files"
    pattern = re.compile(ANNOTATION_FILENAME_REGEX)
    result: dict[int, Path] = {}
    for f in ann_dir.glob("*.csv"):
        m = pattern.match(f.name)
        if m:
            result[int(m.group(1))] = f
    return result


def _load_annotations(ann_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ann_path)
    if SIGNAL_QUALITY_COLUMN in df.columns:
        bsq = df[SIGNAL_QUALITY_COLUMN].astype(str).str.strip().str.lower()
        df = df[~bsq.isin(["true", "1", "yes"])]
    if TARGET_COLUMN in df.columns:
        df = df[~df[TARGET_COLUMN].isin(EXCLUDED_RHYTHM_LABELS)]
    return df.reset_index(drop=True)


def _load_raw_signal(case_id: int) -> np.ndarray | None:
    path = VITALDB_WAVEFORMS_DIR / f"case_{case_id}.npy"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False).astype(np.float64).flatten()


# --- Preprocesamiento por variante --------------------------------------------

def _local_bandpass(window: np.ndarray, fs: int = FS) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
    try:
        return filtfilt(b, a, window)
    except ValueError:
        return window


def _normalize_window(w: np.ndarray) -> np.ndarray:
    std = w.std()
    return (w - w.mean()) / (std if std > 1e-8 else 1e-8)


def _preprocess_signal(signal: np.ndarray, mode: str) -> np.ndarray:
    """
    Preprocesamiento global segun el modo de ablation.

    A (current_pipeline) : interp_nans + bandpass + z-norm global
    B (local_window_only): interp_nans solo
    C (local_bandpass)   : interp_nans solo (bandpass se aplica por ventana)
    """
    sig = interpolate_nans(signal)
    if mode == "current_pipeline":
        sig = bandpass_filter(sig, fs=FS)
        sig = normalize_ecg(sig)
    return sig


def _postprocess_windows(windows: np.ndarray, mode: str) -> np.ndarray:
    """Postprocesamiento a nivel de ventana; bandpass local para modo C."""
    result = np.empty_like(windows, dtype=np.float32)
    for i, w in enumerate(windows):
        if mode == "local_bandpass":
            w = _local_bandpass(w)
        result[i] = _normalize_window(w).astype(np.float32)
    return result


def collect_windows_ablation(
    case_ids: list[int],
    ann_map: dict[int, Path],
    mode: str,
    window_sec: int,
    label_filter: str | None,
    max_per_case: int,
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(window_sec * FS)
    rng = np.random.default_rng(RANDOM_SEED)
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for cid in case_ids:
        if cid not in ann_map:
            continue
        raw = _load_raw_signal(cid)
        if raw is None:
            continue
        ann = _load_annotations(ann_map[cid])
        if ann.empty or BEAT_TIME_COLUMN not in ann.columns:
            continue

        try:
            signal = _preprocess_signal(raw, mode)
        except Exception as exc:
            log.debug("case %d preprocess fallo: %s", cid, exc)
            continue

        try:
            windows, specs = build_windows_for_case(
                signal, ann, case_id=cid,
                fs_hz=FS, window_seconds=float(window_sec), overlap=0.0,
            )
        except Exception as exc:
            log.debug("case %d windowing fallo: %s", cid, exc)
            continue

        if windows.shape[0] == 0:
            continue

        labels = np.array([s.label for s in specs], dtype=object)

        if label_filter is not None:
            mask = labels == label_filter
            windows, labels = windows[mask], labels[mask]

        if windows.shape[0] == 0:
            continue

        if windows.shape[0] > max_per_case:
            idx = rng.choice(windows.shape[0], max_per_case, replace=False)
            idx.sort()
            windows, labels = windows[idx], labels[idx]

        try:
            windows = _postprocess_windows(windows, mode)
        except Exception as exc:
            log.debug("case %d postprocess fallo: %s", cid, exc)
            continue

        valid = np.isfinite(windows).all(axis=1)
        all_X.append(windows[valid])
        all_y.append(labels[valid])

    if not all_X:
        return np.empty((0, n_samples), dtype=np.float32), np.array([], dtype=object)

    return np.vstack(all_X), np.concatenate(all_y)


# --- Mini-autoencoder PyTorch -------------------------------------------------

class MiniAutoencoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        h1, h2 = ABLATION_HIDDEN_DIMS
        b = ABLATION_BOTTLENECK
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, b), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(b, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _train_mini(
    X_train: np.ndarray,
    device: torch.device,
    log: logging.Logger,
) -> tuple[MiniAutoencoder, int]:
    input_dim = X_train.shape[1]
    n = len(X_train)
    n_val = max(1, int(n * ABLATION_VAL_SPLIT))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(n)
    X_tr = torch.from_numpy(X_train[idx[n_val:]])
    X_vl = torch.from_numpy(X_train[idx[:n_val]])

    pin = device.type == "cuda"
    tr_loader = DataLoader(
        TensorDataset(X_tr, X_tr), batch_size=ABLATION_BATCH, shuffle=True, pin_memory=pin
    )
    vl_loader = DataLoader(
        TensorDataset(X_vl, X_vl), batch_size=ABLATION_BATCH * 2, pin_memory=pin
    )

    model = MiniAutoencoder(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ABLATION_LR)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state: dict = {}
    patience = 0
    epochs_done = 0

    for ep in range(1, ABLATION_EPOCHS + 1):
        model.train()
        for xb, _ in tr_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad()
            criterion(model(xb), xb).backward()
            opt.step()

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, _ in vl_loader:
                xb = xb.to(device, non_blocking=True)
                vl_loss += criterion(model(xb), xb).item() * xb.size(0)
        vl_loss /= len(X_vl)
        epochs_done = ep

        if vl_loss < best_val - 1e-7:
            best_val = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= ABLATION_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, epochs_done


@torch.no_grad()
def _recon_errors(model: MiniAutoencoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(X)
    errors: list[np.ndarray] = []
    for start in range(0, len(X), 512):
        xb = tensor[start: start + 512].to(device)
        errors.append(((xb - model(xb)) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errors)


# --- Metricas -----------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray,
    errors: np.ndarray,
    train_errors: np.ndarray,
) -> dict:
    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    y_pred = (errors >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, errors))
    ap = float(average_precision_score(y_true, errors))
    cr = classification_report(
        y_true, y_pred, target_names=["Normal", "Anomalo"],
        output_dict=True, zero_division=0,
    )

    precision_arr, recall_arr, thresh_arr = precision_recall_curve(y_true, errors)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    best_thresh = float(thresh_arr[best_idx]) if best_idx < len(thresh_arr) else threshold
    best_f1_val = float(f1_arr[best_idx])

    anom = cr.get("Anomalo", {})
    return {
        "roc_auc": round(auc, 4),
        "roc_auc_inverted": round(1.0 - auc, 4),
        "average_precision": round(ap, 4),
        "precision_anomaly": round(float(anom.get("precision", 0)), 4),
        "recall_anomaly": round(float(anom.get("recall", 0)), 4),
        "f1_anomaly": round(float(anom.get("f1-score", 0)), 4),
        "accuracy": round(float(cr.get("accuracy", 0)), 4),
        "threshold_p95": round(threshold, 8),
        "best_f1_threshold_diagnostic": round(best_thresh, 8),
        "best_f1_diagnostic": round(best_f1_val, 4),
    }


# --- Ablation principal -------------------------------------------------------

def run_ablation(device: torch.device, log: logging.Logger) -> pd.DataFrame:
    log.info("Descubriendo casos de anotacion...")
    ann_map = _discover_annotation_files()
    all_case_ids = sorted(ann_map.keys())
    log.info("Casos disponibles: %d", len(all_case_ids))

    rng_lim = np.random.default_rng(RANDOM_SEED)
    selected = rng_lim.choice(
        all_case_ids, min(ABLATION_MAX_CASES, len(all_case_ids)), replace=False
    ).tolist()
    log.info("Casos seleccionados: %d", len(selected))

    rng_split = np.random.default_rng(RANDOM_SEED)
    shuffled = rng_split.permutation(selected).tolist()
    n_test = max(1, int(len(shuffled) * TEST_SIZE))
    test_ids: list[int] = shuffled[:n_test]
    train_ids: list[int] = shuffled[n_test:]
    log.info("Train: %d casos  |  Test: %d casos", len(train_ids), len(test_ids))

    total_configs = len(PREPROCESS_MODES) * len(WINDOW_SIZES_SEC)
    results: list[dict] = []
    run_num = 0

    for mode in PREPROCESS_MODES:
        for wsec in WINDOW_SIZES_SEC:
            run_num += 1
            log.info(
                "== [%d/%d] mode=%-22s  window=%ds ==",
                run_num, total_configs, mode, wsec,
            )

            t0 = time.time()
            X_tr, _ = collect_windows_ablation(
                train_ids, ann_map, mode=mode, window_sec=wsec,
                label_filter=NORMAL_LABEL,
                max_per_case=ABLATION_MAX_WINDOWS_PER_CASE,
                log=log,
            )
            X_te, y_te_raw = collect_windows_ablation(
                test_ids, ann_map, mode=mode, window_sec=wsec,
                label_filter=None,
                max_per_case=ABLATION_MAX_WINDOWS_PER_CASE,
                log=log,
            )
            load_time = time.time() - t0

            n_train_n = int(X_tr.shape[0])
            n_test_n = int((y_te_raw == NORMAL_LABEL).sum())
            n_test_a = int((y_te_raw != NORMAL_LABEL).sum())

            log.info(
                "   Ventanas: train_N=%d  test_N=%d  test_A=%d  (carga %.1fs)",
                n_train_n, n_test_n, n_test_a, load_time,
            )

            if n_train_n < 50 or n_test_a < 5:
                log.warning("   Datos insuficientes — config omitida.")
                results.append({
                    "preprocess_mode": mode, "window_seconds": wsec,
                    "roc_auc": None, "roc_auc_inverted": None,
                    "average_precision": None, "precision_anomaly": None,
                    "recall_anomaly": None, "f1_anomaly": None, "accuracy": None,
                    "threshold": None, "best_f1_threshold_diagnostic": None,
                    "best_f1_diagnostic": None,
                    "n_train_normal": n_train_n, "n_test_normal": n_test_n,
                    "n_test_anomaly": n_test_a, "epochs_trained": 0, "train_time_s": None,
                })
                continue

            t_tr = time.time()
            model, epochs_done = _train_mini(X_tr, device, log)
            train_time = time.time() - t_tr

            train_errors = _recon_errors(model, X_tr, device)
            y_te_binary = (y_te_raw != NORMAL_LABEL).astype(np.int32)
            test_errors = _recon_errors(model, X_te, device)

            m = _compute_metrics(y_te_binary, test_errors, train_errors)
            log.info(
                "   AUC=%.4f  inv=%.4f  F1_A=%.4f  [%d ep, %.1fs]",
                m["roc_auc"], m["roc_auc_inverted"], m["f1_anomaly"],
                epochs_done, train_time,
            )

            results.append({
                "preprocess_mode": mode,
                "window_seconds": wsec,
                "roc_auc": m["roc_auc"],
                "roc_auc_inverted": m["roc_auc_inverted"],
                "average_precision": m["average_precision"],
                "precision_anomaly": m["precision_anomaly"],
                "recall_anomaly": m["recall_anomaly"],
                "f1_anomaly": m["f1_anomaly"],
                "accuracy": m["accuracy"],
                "threshold": m["threshold_p95"],
                "best_f1_threshold_diagnostic": m["best_f1_threshold_diagnostic"],
                "best_f1_diagnostic": m["best_f1_diagnostic"],
                "n_train_normal": n_train_n,
                "n_test_normal": n_test_n,
                "n_test_anomaly": n_test_a,
                "epochs_trained": epochs_done,
                "train_time_s": round(train_time, 1),
            })

    ablation_df = pd.DataFrame(results)
    ABLATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(ABLATION_CSV, index=False)
    log.info("Ablation results -> %s", ABLATION_CSV)
    return ablation_df


# =============================================================================
# PARTE 3 - RECOMENDACIONES
# =============================================================================

def print_recommendations(
    diag_df: pd.DataFrame | None,
    ablation_df: pd.DataFrame | None,
) -> None:
    print("\n" + "=" * 65)
    print("RECOMENDACIONES AUTOMATICAS")
    print("=" * 65)

    # A) Score invertido
    print("\n-- A) Score de anomalia " + "-" * 40)
    if EXISTING_RESULTS_CSV.exists():
        df = pd.read_csv(EXISTING_RESULTS_CSV)
        auc_orig = roc_auc_score(df["true_binary"], df["reconstruction_error"])
        if auc_orig < 0.5:
            print(f"  [!] Score INVERTIDO  (AUC actual = {auc_orig:.4f} < 0.50)")
            print("      Las anomalias reconstruyen MEJOR que los normales.")
            print("      -> En produccion: score = -reconstruction_error")
            print("      -> O flag si error < umbral_bajo")
        else:
            print(f"  [OK] Score correcto  (AUC actual = {auc_orig:.4f} >= 0.50)")
    else:
        print("  (CSV de resultados no disponible)")

    if ablation_df is None or ablation_df.empty:
        print("\n  (Ablation no ejecutada — usar --skip-ablation False)")
        print("=" * 65 + "\n")
        return

    valid = ablation_df.dropna(subset=["roc_auc"]).copy()
    if valid.empty:
        print("\n  (Todas las configs del ablation fueron omitidas por datos insuficientes)")
        print("=" * 65 + "\n")
        return

    valid["best_auc"] = valid[["roc_auc", "roc_auc_inverted"]].max(axis=1)
    best = valid.loc[valid["best_auc"].idxmax()]

    # B) Mejor config
    print("\n-- B) Mejor pipeline del ablation " + "-" * 30)
    print(
        f"  Mejor config : preprocess={best['preprocess_mode']}  "
        f"window={best['window_seconds']}s"
    )
    print(
        f"  AUC          = {best['roc_auc']:.4f}  "
        f"(inv={best['roc_auc_inverted']:.4f}  efectivo={best['best_auc']:.4f})"
    )

    print("\n  AUC efectivo por modo de preprocesamiento:")
    by_mode = (
        valid.groupby("preprocess_mode")["best_auc"].max()
        .sort_values(ascending=False)
    )
    for mode, val in by_mode.items():
        marker = "<-- mejor" if mode == best["preprocess_mode"] else ""
        print(f"    {mode:<25} {val:.4f}  {marker}")

    print("\n  AUC efectivo por tamano de ventana:")
    by_window = (
        valid.groupby("window_seconds")["best_auc"].max()
        .sort_values(ascending=False)
    )
    for wsec, val in by_window.items():
        marker = "<-- mejor" if wsec == best["window_seconds"] else ""
        print(f"    {wsec}s   {val:.4f}  {marker}")

    # C) Pipeline actual vs alternativas
    print("\n-- C) Pipeline actual vs alternativas " + "-" * 26)
    curr = valid[valid["preprocess_mode"] == "current_pipeline"]
    others = valid[valid["preprocess_mode"] != "current_pipeline"]
    if not curr.empty and not others.empty:
        curr_best = float(curr["best_auc"].max())
        other_best = float(others["best_auc"].max())
        delta = other_best - curr_best
        if delta > 0.02:
            best_alt = others.loc[others["best_auc"].idxmax()]
            print(
                f"  [!] Pipeline actual ({curr_best:.4f}) es PEOR "
                f"que la mejor alternativa ({other_best:.4f}, delta=+{delta:.4f})"
            )
            print(
                f"      -> Cambiar a: preprocess={best_alt['preprocess_mode']}  "
                f"window={best_alt['window_seconds']}s"
            )
        elif delta > 0:
            print(
                f"  [~] Pipeline actual ({curr_best:.4f}) ligeramente peor "
                f"(delta=+{delta:.4f}). Diferencia marginal."
            )
        else:
            print(
                f"  [OK] Pipeline actual ({curr_best:.4f}) es competitivo "
                f"vs alternativas ({other_best:.4f})."
            )

    # D) Defendibilidad
    print("\n-- D) Defendibilidad para la entrega de curso " + "-" * 18)
    max_eff = float(valid["best_auc"].max())
    if max_eff >= 0.70:
        print(f"  [OK] DEFENDIBLE — AUC efectivo >= 0.70  ({max_eff:.4f})")
        print("       El modelo distingue normales de anomalos con solidez.")
    elif max_eff >= 0.60:
        print(f"  [~] PARCIALMENTE DEFENDIBLE — AUC 0.60-0.70  ({max_eff:.4f})")
        print("       Resultados razonables. Explicar limitaciones: desbalance, AFIB dominante.")
    elif max_eff >= 0.55:
        print(f"  [?] MARGINAL — AUC 0.55-0.60  ({max_eff:.4f})")
        print("       Mejor que azar, pero apenas. Enfatizar argumento metodologico:")
        print("       enfoque no supervisado, dificultad intrinseca del problema.")
    else:
        print(f"  [X] DIFICIL DE DEFENDER — AUC < 0.55  ({max_eff:.4f})")
        print("       Considerar: mas epocas, arquitectura Conv1D, mas datos normales.")

    # E) Proximos pasos
    print("\n-- E) Proximos pasos sugeridos " + "-" * 33)
    print("  1. Invertir score en produccion si AUC < 0.50 (ver A).")
    print("  2. Re-entrenar con la mejor config del ablation (ver B).")
    print("  3. Aumentar epocas (40->80) si AUC sigue bajo.")
    print("  4. Probar arquitectura Conv1D para capturar morfologia local.")
    print("  5. Revisar si AFIB/AFL tiene patron morfologico simple (explica inversion).")
    print("=" * 65 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    device = _resolve_device(args.device)
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Parte 1: Diagnostico
    log.info("=== PARTE 1: Diagnostico de resultados existentes ===")
    diag_df = run_diagnostics(log)

    # Parte 2: Ablation
    ablation_df: pd.DataFrame | None = None
    if not args.skip_ablation:
        log.info(
            "=== PARTE 2: Ablation study (%d configs) ===",
            len(PREPROCESS_MODES) * len(WINDOW_SIZES_SEC),
        )
        log.info(
            "    max_cases=%d  max_windows/caso=%d  epochs=%d",
            ABLATION_MAX_CASES, ABLATION_MAX_WINDOWS_PER_CASE, ABLATION_EPOCHS,
        )
        ablation_df = run_ablation(device, log)

        print("\n-- TABLA ABLATION " + "-" * 56)
        print(
            f"{'Modo':<25} {'Win':>4} {'AUC':>6} {'AUC_inv':>8} "
            f"{'AP':>6} {'F1_A':>6} {'N_tr':>7} {'N_te_A':>7}"
        )
        print("-" * 75)
        for _, r in ablation_df.iterrows():
            def _fmt(v: object, fmt: str = ".4f") -> str:
                return f"{v:{fmt}}" if pd.notna(v) else "   n/a"
            print(
                f"{str(r['preprocess_mode']):<25} {int(r['window_seconds']):>4} "
                f"{_fmt(r['roc_auc']):>6} {_fmt(r['roc_auc_inverted']):>8} "
                f"{_fmt(r['average_precision']):>6} {_fmt(r['f1_anomaly']):>6} "
                f"{int(r['n_train_normal']):>7} {int(r['n_test_anomaly']):>7}"
            )
        print()
    else:
        log.info("Ablation omitida (--skip-ablation).")

    # Parte 3: Recomendaciones
    log.info("=== PARTE 3: Recomendaciones ===")
    print_recommendations(diag_df, ablation_df)

    print("Archivos generados:")
    if DIAG_CSV.exists():
        print(f"  {DIAG_CSV}")
    if ABLATION_CSV.exists():
        print(f"  {ABLATION_CSV}")


if __name__ == "__main__":
    main()
