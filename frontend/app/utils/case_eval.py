"""Utilities for evaluating a VitalDB ECG case from .npy + annotations.

Feature architecture (active tabular model, 30 features):
  - 5 from annotations:  time_second, rr_prev, rr_next, hr_inst_from_rr_prev,
                         position_in_case
  - 25 from metadata.csv: all clinical/surgical fields
  - 0 from ECG waveform: the .npy signal is ONLY used for visualization

This means:
  Case A (known VitalDB case): all 30 features available -> predict + evaluate.
  Case B (external/unknown):   0/30 features available   -> visualize only.
"""

from __future__ import annotations

import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file)
# ---------------------------------------------------------------------------
_UTILS_DIR    = Path(__file__).resolve().parent         # frontend/app/utils/
_APP_DIR      = _UTILS_DIR.parent                       # frontend/app/
_FRONTEND_DIR = _APP_DIR.parent                         # frontend/
PROJECT_ROOT  = _FRONTEND_DIR.parent                    # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANN_DIR       = PROJECT_ROOT / "data" / "raw" / "physionet_annotations" / "Annotation_Files"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "physionet_annotations" / "metadata.csv"
PARQUET_PATH  = PROJECT_ROOT / "data" / "processed" / "filtered_tabular_modeling_dataset.parquet"
WAVEFORMS_DIR = PROJECT_ROOT / "data" / "raw" / "vitaldb_waveforms"

TARGET_FS: float = 500.0
_CASE_PATTERN = re.compile(r"^case[_-]?(\d+)\.npy$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Label-decoding helper
# ---------------------------------------------------------------------------
def _resolve_label_classes(model) -> "np.ndarray | None":
    """Return the string class labels needed to decode integer predictions.

    Tries in order:
      1. model pipeline last step .classes_  — if already strings, return them
      2. Known keys in models/*.json metadata files
      3. Sorted unique rhythm_label values from the parquet dataset (fallback)
    Returns None if no mapping is found.
    """
    import json as _json

    # 1. Model's own classes_ (last step of a Pipeline, or the model itself)
    _clf = model
    if hasattr(model, "steps"):
        try:
            _clf = model.steps[-1][1]
        except Exception:
            pass
    if hasattr(_clf, "classes_"):
        _c = np.asarray(_clf.classes_)
        if _c.dtype.kind in ("U", "O", "S"):   # already strings
            return _c

    # 2. Known keys inside models/*.json
    _CLASS_KEYS = (
        "label_classes", "classes", "target_classes",
        "class_names", "labels", "model_classes",
    )
    _META_PATHS = [
        PROJECT_ROOT / "models" / "model_artifacts_metadata.json",
        PROJECT_ROOT / "models" / "tabular_best_model_metadata.json",
    ]
    for _mp in _META_PATHS:
        if not _mp.exists():
            continue
        try:
            with open(_mp) as _f:
                _meta = _json.load(_f)
            for _key in _CLASS_KEYS:
                if _key in _meta and isinstance(_meta[_key], list) and _meta[_key]:
                    return np.array(_meta[_key])
        except Exception:
            pass

    # 3. Sorted unique labels from the parquet (same order as LabelEncoder.fit)
    if PARQUET_PATH.exists():
        try:
            _df = pd.read_parquet(PARQUET_PATH, columns=["rhythm_label"])
            _classes = sorted(_df["rhythm_label"].dropna().unique().tolist())
            if _classes:
                return np.array(_classes)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# 1. extract_case_id_from_filename
# ---------------------------------------------------------------------------
def extract_case_id_from_filename(filename: str) -> int | None:
    """Returns case_id from 'case_XXXX.npy', None if pattern not matched."""
    m = _CASE_PATTERN.match(Path(filename).name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 2. find_annotation_file
# ---------------------------------------------------------------------------
def find_annotation_file(case_id: int, ann_dir: Path = ANN_DIR) -> Path | None:
    """Returns path to Annotation_file_{case_id}.csv, or None if not found."""
    ann_dir = Path(ann_dir)
    for name in (
        f"Annotation_file_{case_id}.csv",
        f"Annotation_files_{case_id}.csv",
    ):
        p = ann_dir / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# 3. load_npy_signal
# ---------------------------------------------------------------------------
def load_npy_signal(source) -> np.ndarray:
    """Loads .npy and returns a flat float64 array.

    `source` can be a str/Path or an UploadedFile (Streamlit) / BytesIO.
    """
    if isinstance(source, (str, Path)):
        arr = np.load(str(source), allow_pickle=False)
    else:
        raw = source.read() if hasattr(source, "read") else source
        arr = np.load(BytesIO(raw), allow_pickle=False)
    return np.asarray(arr, dtype=float).ravel()


# ---------------------------------------------------------------------------
# 4. summarize_npy_signal
# ---------------------------------------------------------------------------
def summarize_npy_signal(signal: np.ndarray, fs: float = TARGET_FS) -> dict:
    """Returns a summary dict for a 1D signal array."""
    n = len(signal)
    nan_mask = np.isnan(signal)
    nan_count = int(nan_mask.sum())
    valid_idx = np.where(~nan_mask)[0]
    first_valid = int(valid_idx[0]) if len(valid_idx) else None
    return {
        "n_samples":          n,
        "dtype":              str(signal.dtype),
        "duration_s":         round(n / fs, 2),
        "fs_assumed_hz":      fs,
        "nan_count":          nan_count,
        "nan_pct":            round(100.0 * nan_count / n, 2) if n > 0 else 0.0,
        "first_valid_index":  first_valid,
        "first_valid_time_s": round(first_valid / fs, 2) if first_valid is not None else None,
    }


# ---------------------------------------------------------------------------
# 5. find_valid_segments
# ---------------------------------------------------------------------------
def find_valid_segments(
    signal: np.ndarray,
    fs: float = TARGET_FS,
    min_duration_s: float = 5.0,
    max_segments: int = 5,
) -> list[tuple[int, int]]:
    """Returns list of (start_idx, end_idx) for non-NaN segments >= min_duration_s."""
    valid = ~np.isnan(signal)
    min_len = int(min_duration_s * fs)
    segs: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(valid):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            if i - start >= min_len:
                segs.append((start, i - 1))
            in_seg = False
            if len(segs) >= max_segments:
                break
    if in_seg and len(signal) - start >= min_len and len(segs) < max_segments:
        segs.append((start, len(signal) - 1))
    return segs


# ---------------------------------------------------------------------------
# 6. load_case_annotations
# ---------------------------------------------------------------------------
def load_case_annotations(annotation_path: Path) -> pd.DataFrame:
    """Loads annotation CSV and filters out invalid / bad-quality rows.

    Returns DataFrame with columns: time_second, beat_type, rhythm_label,
    bad_signal_quality (and bad_signal_quality_label if present).
    Only rows with valid rhythm_label and good signal quality are kept.
    beat_type is kept for descriptive display ONLY — never used as predictor.
    """
    df = pd.read_csv(annotation_path)

    # Drop missing / invalid rhythm_label (same filters as apply_basic_filters)
    if "rhythm_label" in df.columns:
        df = df.dropna(subset=["rhythm_label"]).copy()
        mask_str = (
            df["rhythm_label"].astype(str).str.strip().str.lower()
            .isin({"nan", "none", "", "noise"})   # "Noise" excluded per project methodology
        )
        df = df.loc[~mask_str].copy()

    # Drop bad signal quality
    if "bad_signal_quality" in df.columns:
        bsq = df["bad_signal_quality"]
        if bsq.dtype == object:
            bsq_bad = bsq.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        else:
            bsq_bad = bsq.astype(bool)
        df = df.loc[~bsq_bad].copy()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. load_case_metadata
# ---------------------------------------------------------------------------
def load_case_metadata(
    case_id: int, metadata_path: Path = METADATA_PATH
) -> pd.Series | None:
    """Returns the metadata row for a given case_id, or None."""
    if not Path(metadata_path).exists():
        return None
    meta = pd.read_csv(metadata_path)
    row = meta.loc[meta["case_id"] == case_id]
    return row.iloc[0] if not row.empty else None


# ---------------------------------------------------------------------------
# 8. build_tabular_features_from_case
# ---------------------------------------------------------------------------
_METADATA_NUMERIC = [
    "analyzed_duration_sec", "total_beats", "caseend", "anestart", "aneend",
    "opstart", "opend", "height", "weight", "bmi",
    "preop_plt", "preop_pt", "preop_k", "preop_alb", "preop_ast", "preop_alt", "preop_cr",
    "tubesize", "intraop_uo", "intraop_crystalloid", "intraop_rocu",
]
_METADATA_CATEGORICAL = ["optype", "iv1", "aline1", "cline1"]


def build_tabular_features_from_case(
    annotations: pd.DataFrame,
    metadata_row: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Builds the feature DataFrame (one row per beat) from annotations + metadata.

    Replicates scripts/02_build_filtered_tabular_modeling_dataset.py:
    _add_within_case_time_features() for a single case.

    Returns (feature_df, missing_features) where feature_df has all columns
    from feature_columns that could be built, plus rhythm_label and beat_type
    (for display). missing_features is the list of requested columns not built.
    """
    df = annotations.copy().sort_values("time_second").reset_index(drop=True)
    times = df["time_second"].to_numpy(dtype=float)
    n = len(times)

    # --- Annotation-derived features ---
    rr_prev = np.full(n, np.nan)
    rr_next = np.full(n, np.nan)
    if n >= 2:
        diffs = np.diff(times)
        rr_prev[1:] = diffs
        rr_next[:-1] = diffs

    with np.errstate(divide="ignore", invalid="ignore"):
        hr_inst = 60.0 / rr_prev
    hr_inst[rr_prev <= 0] = np.nan

    t_min, t_max = times.min(), times.max()
    span = t_max - t_min
    pos = (times - t_min) / span if span > 0 else np.zeros(n)

    df["rr_prev"]              = rr_prev
    df["rr_next"]              = rr_next
    df["hr_inst_from_rr_prev"] = hr_inst
    df["position_in_case"]     = pos

    # --- Metadata-derived features (broadcast to all rows) ---
    for col in _METADATA_NUMERIC + _METADATA_CATEGORICAL:
        if col in metadata_row.index:
            val = metadata_row[col]
            try:
                df[col] = float(val) if col in _METADATA_NUMERIC else val
            except (ValueError, TypeError):
                df[col] = np.nan
        else:
            df[col] = np.nan

    # --- Report missing features ---
    missing = [c for c in feature_columns if c not in df.columns]

    # --- Build output ---
    keep = ["time_second", "rhythm_label"]
    if "beat_type" in df.columns:
        keep.append("beat_type")
    keep += [c for c in feature_columns if c in df.columns]
    return df[keep].copy(), missing


# ---------------------------------------------------------------------------
# 9. get_case_features_from_parquet (fast path)
# ---------------------------------------------------------------------------
def get_case_features_from_parquet(
    case_id: int,
    feature_columns: list[str],
    parquet_path: Path = PARQUET_PATH,
) -> pd.DataFrame | None:
    """Fast path: retrieves pre-computed rows from filtered_tabular_modeling_dataset.parquet.

    Using the parquet guarantees feature consistency with training data.
    Returns None if the case is not present.
    """
    if not Path(parquet_path).exists():
        return None
    df = pd.read_parquet(parquet_path)
    case_df = df.loc[df["case_id"] == case_id].copy()
    if case_df.empty:
        return None
    # Build keep list without duplicates, preserving order
    base = ["case_id", "time_second", "rhythm_label"]
    if "beat_type" in case_df.columns:
        base.append("beat_type")
    seen = set(base)
    for c in feature_columns:
        if c in case_df.columns and c not in seen:
            base.append(c)
            seen.add(c)
    return case_df[base].sort_values("time_second").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 10. predict_case_windows
# ---------------------------------------------------------------------------
def predict_case_windows(
    feature_df: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> np.ndarray:
    """Predicts rhythm_label for each beat row. Returns array of string labels."""
    cols = [c for c in feature_columns if c in feature_df.columns]
    preds = np.asarray(model.predict(feature_df[cols]))

    # If the model was trained on label-encoded integers, decode back to strings
    if len(preds) > 0 and preds.dtype.kind in ("i", "u", "f"):
        classes = _resolve_label_classes(model)
        if classes is not None:
            try:
                preds = np.array([classes[int(p)] for p in preds])
            except (IndexError, ValueError):
                pass

    return preds


# ---------------------------------------------------------------------------
# 11. evaluate_case_predictions
# ---------------------------------------------------------------------------
def evaluate_case_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Returns accuracy and error summary. Never invents ground truth."""
    from sklearn.metrics import accuracy_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Guard: if predictions are still numeric but ground truth is strings,
    # label decoding failed — accuracy cannot be computed reliably.
    if y_pred.dtype.kind in ("i", "u", "f") and y_true.dtype.kind in ("U", "O", "S"):
        return {
            "n_beats":        int(len(y_pred)),
            "n_correct":      None,
            "accuracy":       None,
            "classes_present": [],
            "error_table":    None,
            "warning": (
                "El modelo predijo etiquetas codificadas pero no se encontró "
                "el mapeo de clases en metadata. No se puede calcular accuracy."
            ),
        }

    # Safety: convert both to str before comparison
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    mask = ~np.isin(y_true, ["nan", "None", "none", ""])
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) == 0:
        return {"n_beats": 0, "n_correct": 0, "accuracy": None, "classes_present": [], "error_table": None}

    correct = (yt == yp)
    acc = float(accuracy_score(yt, yp))

    # Top errors
    errors = pd.DataFrame({"real": yt[~correct], "pred": yp[~correct]})
    if not errors.empty:
        error_table = (
            errors.groupby(["real", "pred"]).size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
    else:
        error_table = pd.DataFrame(columns=["real", "pred", "count"])

    return {
        "n_beats":        int(len(yt)),
        "n_correct":      int(correct.sum()),
        "accuracy":       acc,
        "classes_present": sorted(set(yt.tolist())),
        "error_table":    error_table,
    }
