"""
scripts/05_train_autoencoder_anomaly.py

Autoencoder PyTorch para detección binaria de anomalías ECG.

Metodología
-----------
* Entrenamiento : solo ventanas normales (rhythm_label == 'N') de casos train.
* Evaluación    : todas las ventanas de casos test (normales + anómalas).
* Score         : error de reconstrucción MSE por ventana.
* Threshold     : percentil p95 de los errores de train (sin ver test).
* Separación    : por case_id — ningún caso aparece en train y test a la vez.

Uso
---
  .venv\\Scripts\\python.exe scripts/05_train_autoencoder_anomaly.py
  .venv\\Scripts\\python.exe scripts/05_train_autoencoder_anomaly.py --fast
  .venv\\Scripts\\python.exe scripts/05_train_autoencoder_anomaly.py --fast --device cuda
  .venv\\Scripts\\python.exe scripts/05_train_autoencoder_anomaly.py --device cpu

Salidas
-------
  models/autoencoder/ecg_autoencoder.pt
  reports/autoencoder/autoencoder_metrics.json
  reports/autoencoder/autoencoder_test_results.csv
  reports/figures/autoencoder/autoencoder_training_loss.png
  reports/figures/autoencoder/autoencoder_roc_curve.png
  reports/figures/autoencoder/autoencoder_error_distribution.png
  reports/figures/autoencoder/autoencoder_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNOTATION_FILENAME_REGEX,
    BEAT_TIME_COLUMN,
    DEFAULT_ECG_FS_HZ,
    DEFAULT_WINDOW_SECONDS,
    EXCLUDED_RHYTHM_LABELS,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SIGNAL_QUALITY_COLUMN,
    TARGET_COLUMN,
    VITALDB_WAVEFORMS_DIR,
    PHYSIONET_DIR,
)
from src.pipeline import preprocess_ecg
from src.windowing import build_windows_for_case

# ─── check PyTorch ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print(
        "[ERROR] PyTorch no está instalado.\n"
        "  Instalar con:\n"
        "    .venv\\Scripts\\python.exe -m pip install torch\n"
        "  O visitar https://pytorch.org/get-started/locally/ para CUDA."
    )
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN BASE
# Algunos valores se sobreescriben por --fast en parse_args().
# ─────────────────────────────────────────────────────────────────────────────

NORMAL_LABEL: str = "N"
FS: int = DEFAULT_ECG_FS_HZ               # 500 Hz
WINDOW_SEC: float = DEFAULT_WINDOW_SECONDS  # 2.0 s → 1 000 muestras
N_SAMPLES: int = int(WINDOW_SEC * FS)      # 1 000

# Arquitectura
BOTTLENECK_DIM: int = 32
HIDDEN_DIMS: tuple[int, ...] = (512, 128)

# Entrenamiento (modo normal)
EPOCHS: int = 40
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
PATIENCE: int = 5       # early stopping sobre val_loss
VAL_SPLIT: float = 0.10

# Threshold de anomalía
THRESHOLD_PERCENTILE: int = 95

# Split de casos
TEST_SIZE: float = 0.25

# Caps de datos (modo normal)
MAX_WINDOWS_PER_CASE: int | None = 200
MAX_CASES: int | None = None  # None = todos

# --fast overrides
FAST_MAX_CASES: int = 80
FAST_MAX_WINDOWS: int = 100
FAST_EPOCHS: int = 15

# Directorios de salida
AUTOENCODER_MODEL_DIR: Path = MODELS_DIR / "autoencoder"
AUTOENCODER_REPORTS_DIR: Path = REPORTS_DIR / "autoencoder"
AUTOENCODER_FIGURES_DIR: Path = FIGURES_DIR / "autoencoder"


# ─────────────────────────────────────────────────────────────────────────────
# ARGPARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autoencoder PyTorch para detección de anomalías ECG."
    )
    p.add_argument(
        "--fast", action="store_true",
        help=(
            f"Modo rápido: limita a {FAST_MAX_CASES} casos, "
            f"{FAST_MAX_WINDOWS} ventanas/caso y {FAST_EPOCHS} épocas."
        ),
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Dispositivo de cómputo (default: auto).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

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


def _load_signal(case_id: int) -> np.ndarray | None:
    path = VITALDB_WAVEFORMS_DIR / f"case_{case_id}.npy"
    if not path.exists():
        return None
    signal = np.load(path, allow_pickle=False)
    return signal.astype(np.float64).flatten()


def _normalize_windows(X: np.ndarray) -> np.ndarray:
    """Z-score por ventana (cada fila normalizada independientemente)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (X - mean) / std


def collect_windows(
    case_ids: list[int],
    ann_map: dict[int, Path],
    label_filter: str | None,
    max_per_case: int | None,
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae ventanas ECG para una lista de casos.

    Returns
    -------
    X : float32 (n, N_SAMPLES)  — ventanas z-normalizadas por ventana
    y : object  (n,)            — etiqueta de ritmo
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for cid in case_ids:
        if cid not in ann_map:
            continue

        signal = _load_signal(cid)
        if signal is None:
            continue

        ann = _load_annotations(ann_map[cid])
        if ann.empty or BEAT_TIME_COLUMN not in ann.columns:
            continue

        try:
            signal = preprocess_ecg(signal, original_fs=FS)
        except Exception as exc:
            log.debug("case %d — preprocess_ecg falló: %s", cid, exc)
            continue

        try:
            windows, specs = build_windows_for_case(
                signal, ann, case_id=cid,
                fs_hz=FS, window_seconds=WINDOW_SEC, overlap=0.0,
            )
        except Exception as exc:
            log.debug("case %d — build_windows falló: %s", cid, exc)
            continue

        if windows.shape[0] == 0:
            continue

        labels = np.array([s.label for s in specs], dtype=object)

        if label_filter is not None:
            mask = labels == label_filter
            windows = windows[mask]
            labels = labels[mask]

        if windows.shape[0] == 0:
            continue

        if max_per_case is not None and windows.shape[0] > max_per_case:
            idx = rng.choice(windows.shape[0], max_per_case, replace=False)
            idx.sort()
            windows = windows[idx]
            labels = labels[idx]

        all_X.append(windows.astype(np.float32))
        all_y.append(labels)

    if not all_X:
        return np.empty((0, N_SAMPLES), dtype=np.float32), np.array([], dtype=object)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Descartar ventanas con NaN / Inf
    valid = np.isfinite(X).all(axis=1)
    X, y = X[valid], y[valid]

    X = _normalize_windows(X).astype(np.float32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────────────────────

class ECGAutoencoder(nn.Module):
    """
    MLP Autoencoder simétrico.

    Encoder : 1000 → 512 → 128 → 32
    Decoder : 32   → 128 → 512 → 1000
    """

    def __init__(self, input_dim: int = N_SAMPLES,
                 hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
                 bottleneck_dim: int = BOTTLENECK_DIM) -> None:
        super().__init__()

        # ── encoder ──────────────────────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, bottleneck_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        # ── decoder ──────────────────────────────────────────────────────────
        dec_layers: list[nn.Module] = []
        prev = bottleneck_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))  # salida lineal
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    X_train: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    log: logging.Logger,
) -> tuple[ECGAutoencoder, list[float], list[float]]:
    """
    Entrena el autoencoder con early stopping manual.

    Returns
    -------
    model          : mejor modelo (menor val_loss)
    train_losses   : MSE por época (train)
    val_losses     : MSE por época (val)
    """
    # ── Split train / val ────────────────────────────────────────────────────
    n = X_train.shape[0]
    n_val = max(1, int(n * VAL_SPLIT))
    rng = np.random.default_rng(RANDOM_SEED)
    idx_all = rng.permutation(n)
    idx_val = idx_all[:n_val]
    idx_tr = idx_all[n_val:]

    X_tr = torch.from_numpy(X_train[idx_tr])
    X_vl = torch.from_numpy(X_train[idx_val])

    use_pin = device.type == "cuda"
    tr_loader = DataLoader(
        TensorDataset(X_tr, X_tr),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin,
    )
    vl_loader = DataLoader(
        TensorDataset(X_vl, X_vl),
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=use_pin,
    )

    model = ECGAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state: dict = {}
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, _ in tr_loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, xb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(X_tr)

        # ── val ──────────────────────────────────────────────────────────────
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, _ in vl_loader:
                xb = xb.to(device, non_blocking=True)
                out = model(xb)
                vl_loss += criterion(out, xb).item() * xb.size(0)
        vl_loss /= len(X_vl)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        log.info(
            "Época %3d/%d  train_loss=%.6f  val_loss=%.6f",
            epoch, epochs, tr_loss, vl_loss,
        )

        # ── early stopping ───────────────────────────────────────────────────
        if vl_loss < best_val - 1e-7:
            best_val = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping en época %d (patience=%d).", epoch, PATIENCE)
                break

    # Restaurar mejor estado
    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA (errores de reconstrucción)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruction_errors(
    model: ECGAutoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """MSE por ventana entre señal original y reconstruida."""
    model.eval()
    errors: list[np.ndarray] = []
    tensor = torch.from_numpy(X)
    for start in range(0, len(X), batch_size):
        xb = tensor[start : start + batch_size].to(device)
        out = model(xb)
        mse = ((xb - out) ** 2).mean(dim=1).cpu().numpy()
        errors.append(mse)
    return np.concatenate(errors)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────────────────────

def _save_training_loss(
    train_losses: list[float], val_losses: list[float], out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train MSE")
    ax.plot(val_losses, label="Val MSE", linestyle="--")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Pérdida de Entrenamiento — Autoencoder ECG (PyTorch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_training_loss.png", dpi=150)
    plt.close(fig)


def _save_roc_curve(
    y_true: np.ndarray, scores: np.ndarray, auc: float, out_dir: Path
) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC — Detección de Anomalías ECG (Autoencoder)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_roc_curve.png", dpi=150)
    plt.close(fig)


def _save_error_distribution(
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    y_test_binary: np.ndarray,
    threshold: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_errors, bins=80, alpha=0.55, color="steelblue",
            label="Train (solo normales)")
    ax.hist(test_errors[y_test_binary == 0], bins=80, alpha=0.55,
            color="green", label="Test — normales")
    ax.hist(test_errors[y_test_binary == 1], bins=80, alpha=0.55,
            color="tomato", label="Test — anómalos")
    ax.axvline(threshold, color="black", linestyle="--", lw=2,
               label=f"Umbral p{THRESHOLD_PERCENTILE} = {threshold:.5f}")
    ax.set_xlabel("Error de reconstrucción (MSE)")
    ax.set_ylabel("Número de ventanas")
    ax.set_title("Distribución del Error de Reconstrucción")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_error_distribution.png", dpi=150)
    plt.close(fig)


def _save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Anómalo"]
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de Confusión — Autoencoder")
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_confusion_matrix.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Aplicar overrides de --fast
    max_cases = MAX_CASES
    max_windows = MAX_WINDOWS_PER_CASE
    epochs = EPOCHS
    if args.fast:
        max_cases = FAST_MAX_CASES
        max_windows = FAST_MAX_WINDOWS
        epochs = FAST_EPOCHS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ── Dispositivo ───────────────────────────────────────────────────────────
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.error("--device cuda solicitado pero CUDA no está disponible.")
            sys.exit(1)
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)\n")
    else:
        print()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Directorios de salida
    for d in (AUTOENCODER_MODEL_DIR, AUTOENCODER_REPORTS_DIR, AUTOENCODER_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Descubrir casos ────────────────────────────────────────────────────
    log.info("Buscando archivos de anotación...")
    ann_map = _discover_annotation_files()
    all_case_ids = sorted(ann_map.keys())
    log.info("Casos con anotaciones encontrados: %d", len(all_case_ids))

    if max_cases is not None:
        rng_lim = np.random.default_rng(RANDOM_SEED)
        all_case_ids = rng_lim.choice(
            all_case_ids, min(max_cases, len(all_case_ids)), replace=False
        ).tolist()
        log.info(
            "Casos limitados → %d (max_cases=%d, fast=%s)",
            len(all_case_ids), max_cases, args.fast,
        )

    # ── 2. Split train / test por case_id ─────────────────────────────────────
    rng_split = np.random.default_rng(RANDOM_SEED)
    shuffled = rng_split.permutation(all_case_ids).tolist()
    n_test = max(1, int(len(shuffled) * TEST_SIZE))
    test_case_ids: list[int] = shuffled[:n_test]
    train_case_ids: list[int] = shuffled[n_test:]
    log.info("Casos train: %d  |  Casos test: %d", len(train_case_ids), len(test_case_ids))

    # ── 3. Ventanas de entrenamiento (solo normales) ───────────────────────────
    log.info("Extrayendo ventanas de TRAIN (rhythm_label == '%s')...", NORMAL_LABEL)
    t0 = time.time()
    X_train, _ = collect_windows(
        train_case_ids, ann_map,
        label_filter=NORMAL_LABEL,
        max_per_case=max_windows,
        log=log,
    )
    log.info("Ventanas train (normales): %d  [%.1f s]", X_train.shape[0], time.time() - t0)

    if X_train.shape[0] < 100:
        log.error(
            "Muy pocas ventanas de entrenamiento (%d). "
            "Verificar rutas de datos o reducir --fast.",
            X_train.shape[0],
        )
        sys.exit(1)

    # ── 4. Ventanas de test (todas las etiquetas) ─────────────────────────────
    log.info("Extrayendo ventanas de TEST (todas las etiquetas)...")
    t0 = time.time()
    X_test, y_test_raw = collect_windows(
        test_case_ids, ann_map,
        label_filter=None,
        max_per_case=max_windows,
        log=log,
    )
    log.info("Ventanas test (total): %d  [%.1f s]", X_test.shape[0], time.time() - t0)

    if X_test.shape[0] == 0:
        log.error("Sin ventanas de test. Abortar.")
        sys.exit(1)

    y_test_binary = (y_test_raw != NORMAL_LABEL).astype(np.int32)
    n_normal = int((y_test_binary == 0).sum())
    n_anomal = int((y_test_binary == 1).sum())
    log.info("Test — Normales: %d  |  Anómalos: %d", n_normal, n_anomal)

    if n_anomal == 0:
        log.error("No hay ventanas anómalas en el test. ROC-AUC indefinido.")
        sys.exit(1)

    # ── 5. Entrenamiento ──────────────────────────────────────────────────────
    log.info(
        "Entrenando autoencoder PyTorch (%d épocas, batch=%d, device=%s)...",
        epochs, BATCH_SIZE, device,
    )
    t_train = time.time()
    model, train_losses, val_losses = train_autoencoder(
        X_train, device, epochs, BATCH_SIZE, log
    )
    train_time = time.time() - t_train
    log.info("Entrenamiento completado en %.1f s", train_time)

    # ── 6. Errores de reconstrucción ──────────────────────────────────────────
    log.info("Calculando errores de reconstrucción...")
    train_errors = reconstruction_errors(model, X_train, device)
    test_errors = reconstruction_errors(model, X_test, device)

    # ── 7. Threshold ──────────────────────────────────────────────────────────
    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    log.info("Umbral p%d de train = %.6f", THRESHOLD_PERCENTILE, threshold)

    y_pred_binary = (test_errors >= threshold).astype(np.int32)

    # ── 8. Métricas ───────────────────────────────────────────────────────────
    auc = roc_auc_score(y_test_binary, test_errors)
    report_dict = classification_report(
        y_test_binary, y_pred_binary,
        target_names=["Normal", "Anómalo"],
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_test_binary, y_pred_binary,
        target_names=["Normal", "Anómalo"],
        zero_division=0,
    )
    log.info("ROC-AUC = %.4f", auc)
    log.info("\n%s", report_str)

    # ── 9. Guardar métricas JSON ──────────────────────────────────────────────
    epochs_done = len(train_losses)
    metrics = {
        "approach": "autoencoder_anomaly_detection_pytorch",
        "framework": "pytorch",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "roc_auc": round(float(auc), 4),
        "threshold": round(threshold, 8),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "train_windows_normal": int(X_train.shape[0]),
        "test_windows_total": int(X_test.shape[0]),
        "test_windows_normal": n_normal,
        "test_windows_anomalous": n_anomal,
        "train_cases": len(train_case_ids),
        "test_cases": len(test_case_ids),
        "epochs_configured": epochs,
        "epochs_trained": epochs_done,
        "bottleneck_dim": BOTTLENECK_DIM,
        "window_seconds": WINDOW_SEC,
        "fs_hz": FS,
        "fast_mode": args.fast,
        "train_time_seconds": round(train_time, 1),
        "classification_report": report_dict,
    }
    metrics_path = AUTOENCODER_REPORTS_DIR / "autoencoder_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Métricas → %s", metrics_path)

    # ── 10. Guardar resultados CSV ────────────────────────────────────────────
    results_df = pd.DataFrame({
        "true_label": y_test_raw,
        "true_binary": y_test_binary,
        "reconstruction_error": test_errors.astype(np.float64),
        "pred_binary": y_pred_binary,
    })
    results_path = AUTOENCODER_REPORTS_DIR / "autoencoder_test_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Resultados CSV → %s", results_path)

    # ── 11. Guardar modelo .pt ────────────────────────────────────────────────
    model_path = AUTOENCODER_MODEL_DIR / "ecg_autoencoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "bottleneck_dim": BOTTLENECK_DIM,
            "hidden_dims": HIDDEN_DIMS,
            "n_samples": N_SAMPLES,
            "fs_hz": FS,
            "window_seconds": WINDOW_SEC,
        },
        model_path,
    )
    log.info("Modelo → %s", model_path)

    # ── 12. Figuras ───────────────────────────────────────────────────────────
    log.info("Generando figuras...")
    _save_training_loss(train_losses, val_losses, AUTOENCODER_FIGURES_DIR)
    _save_roc_curve(y_test_binary, test_errors, auc, AUTOENCODER_FIGURES_DIR)
    _save_error_distribution(
        train_errors, test_errors, y_test_binary, threshold, AUTOENCODER_FIGURES_DIR
    )
    _save_confusion_matrix(y_test_binary, y_pred_binary, AUTOENCODER_FIGURES_DIR)
    log.info("Figuras → %s", AUTOENCODER_FIGURES_DIR)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("AUTOENCODER PyTorch — RESULTADOS FINALES")
    print("=" * 62)
    print(f"  Device                : {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  ROC-AUC               : {auc:.4f}")
    print(f"  Umbral (p{THRESHOLD_PERCENTILE} train)  : {threshold:.6f}")
    print(f"  Ventanas train normal : {X_train.shape[0]:,}")
    print(f"  Ventanas test total   : {X_test.shape[0]:,}  "
          f"(N={n_normal:,} / Anóm={n_anomal:,})")
    print(f"  Épocas entrenadas     : {epochs_done}/{epochs}")
    print(f"  Tiempo entrenamiento  : {train_time:.1f} s")
    print("-" * 62)
    print(report_str)
    print("=" * 62)
    print(f"Modelo   → {model_path}")
    print(f"Métricas → {metrics_path}")
    print(f"Figuras  → {AUTOENCODER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
