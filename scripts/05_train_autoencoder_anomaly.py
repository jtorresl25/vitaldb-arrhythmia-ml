"""
scripts/05_train_autoencoder_anomaly.py

Autoencoder para detección binaria de anomalías ECG.

Metodología
-----------
* Entrenamiento : solo ventanas normales (rhythm_label == 'N') de casos de
  entrenamiento.
* Evaluación    : todas las ventanas de casos de test (normales + anómalas).
* Score         : error de reconstrucción MSE por ventana.
* Threshold     : percentil p95 de los errores de train (no depende del test).
* Separación    : por case_id — ningún caso aparece en train y test.

Salidas
-------
models/autoencoder/ecg_autoencoder.keras
reports/autoencoder/autoencoder_metrics.json
reports/autoencoder/autoencoder_test_results.csv
reports/figures/autoencoder/  (4 figuras PNG)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── project root ────────────────────────────────────────────────────────────
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

# ─── check tensorflow ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print(
        "[ERROR] TensorFlow no está instalado.\n"
        "  Instalar con:  pip install tensorflow\n"
        "  o              conda install -c conda-forge tensorflow"
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
# CONFIGURACIÓN  (ajustar para acelerar o profundizar)
# ─────────────────────────────────────────────────────────────────────────────

NORMAL_LABEL: str = "N"
FS: int = DEFAULT_ECG_FS_HZ              # 500 Hz
WINDOW_SEC: float = DEFAULT_WINDOW_SECONDS  # 2.0 s → 1 000 muestras/ventana
N_SAMPLES: int = int(WINDOW_SEC * FS)    # 1 000

# Arquitectura del autoencoder
BOTTLENECK_DIM: int = 32
HIDDEN_DIMS: tuple[int, ...] = (512, 128)  # idéntico en encoder y decoder

# Entrenamiento
EPOCHS: int = 40
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3

# Threshold de anomalía: percentil de errores de train
THRESHOLD_PERCENTILE: int = 95

# Fracción de casos reservada para test
TEST_SIZE: float = 0.25

# Cap de ventanas por caso (None = sin límite; reducir para rapidez)
MAX_WINDOWS_PER_CASE: int | None = 200

# Límite total de casos a procesar (None = todos)
# Poner e.g. 80 para una prueba rápida de ~20 min en CPU
MAX_CASES: int | None = None

# Directorios de salida
AUTOENCODER_MODEL_DIR: Path = MODELS_DIR / "autoencoder"
AUTOENCODER_REPORTS_DIR: Path = REPORTS_DIR / "autoencoder"
AUTOENCODER_FIGURES_DIR: Path = FIGURES_DIR / "autoencoder"


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def _discover_annotation_files() -> dict[int, Path]:
    """Devuelve {case_id: Path} para todos los archivos de anotación."""
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
    """Carga un CSV de anotaciones aplicando filtros estándar."""
    df = pd.read_csv(ann_path)

    # Filtrar mala calidad de señal (columna puede ser bool o string)
    if SIGNAL_QUALITY_COLUMN in df.columns:
        bsq = df[SIGNAL_QUALITY_COLUMN].astype(str).str.strip().str.lower()
        df = df[~bsq.isin(["true", "1", "yes"])]

    # Filtrar etiquetas excluidas (p.ej. Noise)
    if TARGET_COLUMN in df.columns:
        df = df[~df[TARGET_COLUMN].isin(EXCLUDED_RHYTHM_LABELS)]

    return df.reset_index(drop=True)


def _load_signal(case_id: int) -> np.ndarray | None:
    """Carga la señal ECG .npy del caso. Retorna None si no existe."""
    path = VITALDB_WAVEFORMS_DIR / f"case_{case_id}.npy"
    if not path.exists():
        return None
    signal = np.load(path, allow_pickle=False)
    return signal.astype(np.float64).flatten()


def _normalize_windows(X: np.ndarray) -> np.ndarray:
    """Z-score por ventana (cada fila de X normalizada independientemente)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (X - mean) / std


def collect_windows(
    case_ids: list[int],
    ann_map: dict[int, Path],
    label_filter: str | None = None,
    max_per_case: int | None = MAX_WINDOWS_PER_CASE,
    log: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga señales ECG, preprocesa y extrae ventanas para una lista de casos.

    Parameters
    ----------
    case_ids      : lista de case_id a procesar
    ann_map       : case_id → Path de anotaciones
    label_filter  : si no es None, conserva solo ventanas con esa etiqueta
    max_per_case  : máximo de ventanas por caso (None = sin límite)

    Returns
    -------
    X : float32 array (n, N_SAMPLES) — ventanas z-normalizadas por ventana
    y : object array (n,)            — etiqueta de ritmo por ventana
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

        # Preprocesamiento de la señal completa (filtro paso-banda + z-norm)
        try:
            signal = preprocess_ecg(signal, original_fs=FS)
        except Exception as exc:
            if log:
                log.debug("case %d — preprocess_ecg falló: %s", cid, exc)
            continue

        # Extracción de ventanas centradas en cada latido
        try:
            windows, specs = build_windows_for_case(
                signal, ann, case_id=cid,
                fs_hz=FS, window_seconds=WINDOW_SEC, overlap=0.0,
            )
        except Exception as exc:
            if log:
                log.debug("case %d — build_windows falló: %s", cid, exc)
            continue

        if windows.shape[0] == 0:
            continue

        # Etiquetas desde los WindowSpec
        labels = np.array([s.label for s in specs], dtype=object)

        # Filtro por etiqueta si se solicitó
        if label_filter is not None:
            mask = labels == label_filter
            windows = windows[mask]
            labels = labels[mask]

        if windows.shape[0] == 0:
            continue

        # Cap de ventanas por caso — muestreo aleatorio sin reemplazo
        if max_per_case is not None and windows.shape[0] > max_per_case:
            idx = rng.choice(windows.shape[0], max_per_case, replace=False)
            idx.sort()
            windows = windows[idx]
            labels = labels[idx]

        all_X.append(windows.astype(np.float32))
        all_y.append(labels)

    if not all_X:
        return np.empty((0, N_SAMPLES), dtype=np.float32), np.array([], dtype=object)

    X = np.vstack(all_X)           # (n, N_SAMPLES)
    y = np.concatenate(all_y)      # (n,)

    # Descartar ventanas con NaN o Inf (p.ej. segmentos corruptos)
    valid = np.isfinite(X).all(axis=1)
    X, y = X[valid], y[valid]

    # Normalización z-score por ventana
    X = _normalize_windows(X)

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITECTURA DEL AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────

def build_autoencoder(input_dim: int = N_SAMPLES) -> keras.Model:
    """
    Dense autoencoder simétrico.

    Encoder : input_dim → 512 → 128 → BOTTLENECK_DIM
    Decoder : BOTTLENECK_DIM → 128 → 512 → input_dim
    """
    inputs = keras.Input(shape=(input_dim,), name="ecg_input")

    # Encoder
    x = inputs
    for units in HIDDEN_DIMS:
        x = keras.layers.Dense(units, activation="relu")(x)
    bottleneck = keras.layers.Dense(BOTTLENECK_DIM, activation="relu",
                                    name="bottleneck")(x)

    # Decoder (espejo)
    x = bottleneck
    for units in reversed(HIDDEN_DIMS):
        x = keras.layers.Dense(units, activation="relu")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear",
                                 name="reconstruction")(x)

    model = keras.Model(inputs, outputs, name="ecg_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────────────────────

def _save_training_loss(history: keras.callbacks.History, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train MSE")
    ax.plot(history.history["val_loss"], label="Val MSE", linestyle="--")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Pérdida de Entrenamiento — Autoencoder ECG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_training_loss.png", dpi=150)
    plt.close(fig)


def _save_roc_curve(y_true: np.ndarray, scores: np.ndarray,
                    auc: float, out_dir: Path) -> None:
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


def _save_error_distribution(train_errors: np.ndarray, test_errors: np.ndarray,
                              y_test_binary: np.ndarray, threshold: float,
                              out_dir: Path) -> None:
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


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            out_dir: Path) -> None:
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # Reproducibilidad
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Crear directorios de salida
    for d in (AUTOENCODER_MODEL_DIR, AUTOENCODER_REPORTS_DIR, AUTOENCODER_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Descubrir casos disponibles ────────────────────────────────────────
    log.info("Buscando archivos de anotación...")
    ann_map = _discover_annotation_files()
    all_case_ids = sorted(ann_map.keys())
    log.info("Casos con anotaciones: %d", len(all_case_ids))

    if MAX_CASES is not None:
        rng_global = np.random.default_rng(RANDOM_SEED)
        all_case_ids = rng_global.choice(
            all_case_ids, min(MAX_CASES, len(all_case_ids)), replace=False
        ).tolist()
        log.info("Casos limitados por MAX_CASES=%d → %d casos", MAX_CASES, len(all_case_ids))

    # ── 2. Split train / test por case_id ─────────────────────────────────────
    rng_split = np.random.default_rng(RANDOM_SEED)
    shuffled = rng_split.permutation(all_case_ids).tolist()
    n_test = max(1, int(len(shuffled) * TEST_SIZE))
    test_case_ids: list[int] = shuffled[:n_test]
    train_case_ids: list[int] = shuffled[n_test:]
    log.info("Casos train: %d  |  Casos test: %d", len(train_case_ids), len(test_case_ids))

    # ── 3. Ventanas de entrenamiento (solo normales) ───────────────────────────
    log.info("Extrayendo ventanas de TRAIN (solo rhythm_label == '%s')...", NORMAL_LABEL)
    t_load = time.time()
    X_train, _ = collect_windows(
        train_case_ids, ann_map,
        label_filter=NORMAL_LABEL,
        max_per_case=MAX_WINDOWS_PER_CASE,
        log=log,
    )
    log.info(
        "Ventanas de train (normales): %d  [%.1f s]",
        X_train.shape[0], time.time() - t_load,
    )

    if X_train.shape[0] < 100:
        log.error(
            "Muy pocas ventanas de entrenamiento (%d). "
            "Verificar rutas de datos o aumentar MAX_WINDOWS_PER_CASE.",
            X_train.shape[0],
        )
        sys.exit(1)

    # ── 4. Ventanas de test (todas las etiquetas) ─────────────────────────────
    log.info("Extrayendo ventanas de TEST (todas las etiquetas)...")
    t_load = time.time()
    X_test, y_test_raw = collect_windows(
        test_case_ids, ann_map,
        label_filter=None,
        max_per_case=MAX_WINDOWS_PER_CASE,
        log=log,
    )
    log.info(
        "Ventanas de test (total): %d  [%.1f s]",
        X_test.shape[0], time.time() - t_load,
    )

    if X_test.shape[0] == 0:
        log.error("Sin ventanas de test. Abortar.")
        sys.exit(1)

    # Etiquetas binarias: 0 = normal, 1 = anómalo
    y_test_binary = (y_test_raw != NORMAL_LABEL).astype(np.int32)
    n_normal = int((y_test_binary == 0).sum())
    n_anomal = int((y_test_binary == 1).sum())
    log.info("Test — Normales: %d  |  Anómalos: %d", n_normal, n_anomal)

    if n_anomal == 0:
        log.error("No hay ventanas anómalas en el test. ROC-AUC indefinido.")
        sys.exit(1)

    # ── 5. Construir y entrenar autoencoder ───────────────────────────────────
    log.info("Construyendo autoencoder (bottleneck=%d)...", BOTTLENECK_DIM)
    model = build_autoencoder(N_SAMPLES)
    model.summary(print_fn=log.info)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    log.info("Entrenando %d épocas (batch=%d)...", EPOCHS, BATCH_SIZE)
    t_train = time.time()
    history = model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.10,
        callbacks=[early_stop],
        verbose=1,
    )
    train_time = time.time() - t_train
    log.info("Entrenamiento completado en %.1f s", train_time)

    # ── 6. Errores de reconstrucción ─────────────────────────────────────────
    log.info("Calculando errores de reconstrucción...")
    X_train_recon = model.predict(X_train, batch_size=512, verbose=0)
    train_errors: np.ndarray = np.mean((X_train - X_train_recon) ** 2, axis=1)

    X_test_recon = model.predict(X_test, batch_size=512, verbose=0)
    test_errors: np.ndarray = np.mean((X_test - X_test_recon) ** 2, axis=1)

    # ── 7. Umbral de anomalía ─────────────────────────────────────────────────
    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    log.info(
        "Umbral p%d de errores de train = %.6f", THRESHOLD_PERCENTILE, threshold
    )

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
    epochs_done = len(history.history["loss"])
    metrics = {
        "approach": "autoencoder_anomaly_detection",
        "roc_auc": round(float(auc), 4),
        "threshold": round(threshold, 8),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "train_windows_normal": int(X_train.shape[0]),
        "test_windows_total": int(X_test.shape[0]),
        "test_windows_normal": n_normal,
        "test_windows_anomalous": n_anomal,
        "train_cases": len(train_case_ids),
        "test_cases": len(test_case_ids),
        "epochs_configured": EPOCHS,
        "epochs_trained": epochs_done,
        "bottleneck_dim": BOTTLENECK_DIM,
        "window_seconds": WINDOW_SEC,
        "fs_hz": FS,
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

    # ── 11. Guardar modelo ────────────────────────────────────────────────────
    model_path = AUTOENCODER_MODEL_DIR / "ecg_autoencoder.keras"
    model.save(model_path)
    log.info("Modelo → %s", model_path)

    # ── 12. Figuras ───────────────────────────────────────────────────────────
    log.info("Generando figuras...")
    _save_training_loss(history, AUTOENCODER_FIGURES_DIR)
    _save_roc_curve(y_test_binary, test_errors, auc, AUTOENCODER_FIGURES_DIR)
    _save_error_distribution(
        train_errors, test_errors, y_test_binary, threshold, AUTOENCODER_FIGURES_DIR
    )
    _save_confusion_matrix(y_test_binary, y_pred_binary, AUTOENCODER_FIGURES_DIR)
    log.info("Figuras → %s", AUTOENCODER_FIGURES_DIR)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AUTOENCODER — RESULTADOS FINALES")
    print("=" * 60)
    print(f"  ROC-AUC               : {auc:.4f}")
    print(f"  Umbral (p{THRESHOLD_PERCENTILE} train)  : {threshold:.6f}")
    print(f"  Ventanas train normal : {X_train.shape[0]:,}")
    print(f"  Ventanas test total   : {X_test.shape[0]:,}  "
          f"(N={n_normal:,} / Anóm={n_anomal:,})")
    print(f"  Épocas entrenadas     : {epochs_done}/{EPOCHS}")
    print(f"  Tiempo de entrenamiento: {train_time:.1f} s")
    print("-" * 60)
    print(report_str)
    print("=" * 60)
    print(f"Modelo   → {model_path}")
    print(f"Métricas → {metrics_path}")
    print(f"Figuras  → {AUTOENCODER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
