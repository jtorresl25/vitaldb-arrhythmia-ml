"""ECG visualization components for Streamlit (Plotly, dark theme).

All functions return go.Figure objects; callers pass them to st.plotly_chart().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts import apply_dark_layout

_RHYTHM_COLORS: dict[str, str] = {
    "N":                           "#8a98b5",
    "AFIB/AFL":                    "#f472b6",
    "Patterned Ventricular Ectopy":"#fb923c",
    "Patterned Atrial Ectopy":     "#a78bfa",
    "SVTA":                        "#4a8cff",
    "VT":                          "#f87171",
    "SND":                         "#34d399",
    "AVB":                         "#facc15",
    "WAP/MAT":                     "#94a3b8",
    "Unclassifiable":              "#5d6c8c",
}


def _ecg_trace(
    signal: np.ndarray,
    t_axis: np.ndarray,
    line_color: str = "#2dd4bf",
    fill_color: str = "rgba(45,212,191,0.04)",
    name: str = "ECG",
) -> go.Scatter:
    return go.Scatter(
        x=t_axis,
        y=signal,
        mode="lines",
        name=name,
        line=dict(color=line_color, width=1.1),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="t=%{x:.3f} s  amp=%{y:.4f}<extra></extra>",
    )


def _time_axis(n_samples: int, fs: float, offset: float = 0.0) -> np.ndarray:
    return offset + np.arange(n_samples) / fs


def _dark_axes() -> dict:
    return dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.04)",
        zeroline=False,
        tickfont=dict(size=9),
    )


# ---------------------------------------------------------------------------
# plot_ecg_signal
# ---------------------------------------------------------------------------
def plot_ecg_signal(
    signal: np.ndarray,
    fs: float,
    title: str = "Señal ECG",
    max_seconds: float = 10.0,
    start_time_offset: float = 0.0,
    line_color: str = "#2dd4bf",
    fill_color: str = "rgba(45,212,191,0.04)",
    height: int = 230,
) -> go.Figure:
    """Plots up to max_seconds of the signal starting from start_time_offset."""
    sig = np.asarray(signal, dtype=float).ravel()
    n = min(len(sig), int(max_seconds * fs))
    t = _time_axis(n, fs, offset=start_time_offset)
    fig = go.Figure(_ecg_trace(sig[:n], t, line_color, fill_color))
    apply_dark_layout(
        fig,
        title=dict(text=title, font=dict(size=12, color="#c8d2e5")),
        height=height,
        margin=dict(l=44, r=10, t=36, b=32),
        xaxis=dict(**_dark_axes(), title=dict(text="tiempo (s)", font=dict(size=10))),
        yaxis=dict(**_dark_axes(), title=dict(text="amplitud", font=dict(size=10))),
    )
    return fig


# ---------------------------------------------------------------------------
# plot_raw_vs_processed
# ---------------------------------------------------------------------------
def plot_raw_vs_processed(
    raw_signal: np.ndarray,
    processed_signal: np.ndarray,
    raw_fs: float,
    processed_fs: float = 500.0,
    max_seconds: float = 10.0,
    start_time_offset: float = 0.0,
    height: int = 230,
) -> tuple[go.Figure, go.Figure]:
    """Returns (raw_fig, processed_fig)."""
    fig_raw = plot_ecg_signal(
        raw_signal, raw_fs,
        title="Señal cruda",
        max_seconds=max_seconds,
        start_time_offset=start_time_offset,
        line_color="#2dd4bf",
        fill_color="rgba(45,212,191,0.04)",
        height=height,
    )
    fig_proc = plot_ecg_signal(
        processed_signal, processed_fs,
        title="Señal procesada · filtrada 0.5–40 Hz · normalizada Z-score",
        max_seconds=max_seconds,
        start_time_offset=start_time_offset,
        line_color="#4a8cff",
        fill_color="rgba(74,140,255,0.04)",
        height=height,
    )
    return fig_raw, fig_proc


# ---------------------------------------------------------------------------
# plot_annotations_on_ecg
# ---------------------------------------------------------------------------
def plot_annotations_on_ecg(
    signal: np.ndarray,
    annotations: pd.DataFrame | None,
    fs: float = 500.0,
    max_seconds: float = 10.0,
    start_time_offset: float = 0.0,
    height: int = 310,
) -> go.Figure:
    """ECG trace with vertical dashed lines at annotation beat timestamps.

    Lines are colored by rhythm_label. beat_type is shown in hover (descriptive only).
    """
    sig = np.asarray(signal, dtype=float).ravel()
    n = min(len(sig), int(max_seconds * fs))
    t = _time_axis(n, fs, offset=start_time_offset)
    t_end = float(t[-1]) if len(t) else start_time_offset + max_seconds

    fig = go.Figure(_ecg_trace(sig[:n], t, name="ECG"))

    if annotations is not None and not annotations.empty and "time_second" in annotations.columns:
        ann_win = annotations[
            (annotations["time_second"] >= start_time_offset) &
            (annotations["time_second"] <= t_end)
        ]
        if not ann_win.empty:
            valid = sig[:n]
            y_lo = float(np.nanmin(valid)) if np.any(~np.isnan(valid)) else -1.0
            y_hi = float(np.nanmax(valid)) if np.any(~np.isnan(valid)) else 1.0

            for _, beat in ann_win.iterrows():
                t_beat  = float(beat["time_second"])
                label   = str(beat.get("rhythm_label", "N"))
                btype   = str(beat.get("beat_type", ""))
                color   = _RHYTHM_COLORS.get(label, "#5d6c8c")
                fig.add_shape(
                    type="line",
                    x0=t_beat, x1=t_beat, y0=y_lo, y1=y_hi,
                    line=dict(color=color, width=0.9, dash="dot"),
                )
                # Invisible scatter for hover
                fig.add_trace(go.Scatter(
                    x=[t_beat], y=[(y_lo + y_hi) / 2],
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.6),
                    name=label,
                    showlegend=False,
                    hovertemplate=(
                        f"t={t_beat:.3f} s<br>"
                        f"ritmo: <b>{label}</b><br>"
                        f"beat_type: {btype}<extra></extra>"
                    ),
                ))

    apply_dark_layout(
        fig,
        title=dict(text="ECG + latidos anotados (colores por rhythm_label)", font=dict(size=12, color="#c8d2e5")),
        height=height,
        margin=dict(l=44, r=10, t=36, b=32),
        xaxis=dict(**_dark_axes(), title=dict(text="tiempo (s)", font=dict(size=10))),
        yaxis=dict(**_dark_axes()),
    )
    return fig


# ---------------------------------------------------------------------------
# Binary prediction helpers
# ---------------------------------------------------------------------------
_BINARY_NORMAL_STRS = frozenset({"normal", "n", "0", "false"})


def _is_abnormal_pred(x) -> bool:
    """Return True if x represents an abnormal binary prediction."""
    s = str(x).strip().lower()
    return bool(s) and s not in _BINARY_NORMAL_STRS and s != "nan"


def _is_binary_pred_col(series: pd.Series) -> bool:
    """True if all non-null values in the series are binary labels."""
    unique = {str(v).strip().lower() for v in series.dropna() if str(v).strip() not in ("", "nan")}
    if not unique:
        return False
    return unique.issubset({"normal", "abnormal", "anormal", "0", "1", "n", "false", "true"})


def _rhythm_label_to_binary(label: str) -> str:
    """Convert an original rhythm_label to binary ('normal' / 'abnormal')."""
    return "normal" if str(label).strip() == "N" else "abnormal"


# ---------------------------------------------------------------------------
# plot_ecg_with_prediction_regions
# ---------------------------------------------------------------------------

# Colores de las regiones anómalas (N no se pinta)
_REGION_COLORS: dict[str, str] = {
    "AFIB/AFL":                    "rgba(244,114,182,0.35)",
    "Patterned Ventricular Ectopy":"rgba(251,146,60,0.40)",
    "Patterned Atrial Ectopy":     "rgba(167,139,250,0.38)",
    "SVTA":                        "rgba(74,140,255,0.38)",
    "VT":                          "rgba(248,113,113,0.45)",
    "SND":                         "rgba(52,211,153,0.38)",
    "AVB":                         "rgba(250,204,21,0.42)",
    "WAP/MAT":                     "rgba(148,163,184,0.35)",
    "Unclassifiable":              "rgba(93,108,140,0.35)",
}
_REGION_BORDER: dict[str, str] = {
    "AFIB/AFL":                    "rgba(244,114,182,0.70)",
    "Patterned Ventricular Ectopy":"rgba(251,146,60,0.80)",
    "Patterned Atrial Ectopy":     "rgba(167,139,250,0.75)",
    "SVTA":                        "rgba(74,140,255,0.75)",
    "VT":                          "rgba(248,113,113,0.85)",
    "SND":                         "rgba(52,211,153,0.75)",
    "AVB":                         "rgba(250,204,21,0.80)",
    "WAP/MAT":                     "rgba(148,163,184,0.70)",
    "Unclassifiable":              "rgba(93,108,140,0.70)",
}
_REGION_DEFAULT      = "rgba(93,108,140,0.30)"
_REGION_DEFAULT_BORDER = "rgba(93,108,140,0.60)"
_NORMAL_LABEL = "N"


def plot_ecg_with_prediction_regions(
    signal: np.ndarray,
    fs: float,
    predictions_df: pd.DataFrame,
    time_col: str = "time_second",
    pred_col: str = "prediccion",
    real_col: str = "rhythm_label",
    beat_type_col: str = "beat_type",
    max_seconds: float = 30.0,
    start_offset: float = 0.0,
    title: str = "ECG con regiones predichas por el modelo",
    height: int = 400,
) -> go.Figure:
    """ECG trace coloreado según la predicción binaria del modelo.

    Modo binario (pred_col = "normal" / "abnormal"):
      - Tramos normales: línea gris clara (#cbd5e1, grosor 1.2).
      - Tramos anormales: línea roja (#ef4444, grosor 1.8).
      - Marcadores ✕ donde real (binarizado) ≠ predicción.

    Modo multiclase (pred_col = rhythm labels "N", "AFIB/AFL", …):
      - Franjas de color de fondo por ritmo predicho (comportamiento anterior).
      - Línea ECG gris uniforme.
    """
    # ── Señal recortada ────────────────────────────────────────────────────
    sig = np.asarray(signal, dtype=float).ravel()
    n_disp    = min(len(sig), int(max_seconds * fs))
    t_arr     = _time_axis(n_disp, fs, offset=start_offset)
    t_end     = float(t_arr[-1])
    valid_sig = sig[:n_disp]

    sig_vals  = valid_sig[~np.isnan(valid_sig)]
    y_max     = float(np.max(sig_vals))  if len(sig_vals) else  1.0
    y_min     = float(np.min(sig_vals))  if len(sig_vals) else -1.0
    y_span    = y_max - y_min if y_max > y_min else 1.0
    marker_y  = y_max + y_span * 0.08

    # ── Filtrar y ordenar predicciones en la ventana visible ───────────────
    if predictions_df is None or predictions_df.empty or time_col not in predictions_df.columns:
        # Fallback: plain gray ECG, no predictions
        fig = go.Figure(go.Scatter(
            x=t_arr, y=valid_sig, mode="lines",
            name="ECG", line=dict(color="#d1d5db", width=1.1),
            hovertemplate="t=%{x:.3f} s  amp=%{y:.4f}<extra></extra>",
            showlegend=False,
        ))
        apply_dark_layout(
            fig,
            title=dict(text=title, font=dict(size=12, color="#c8d2e5")),
            height=height,
            margin=dict(l=44, r=10, t=40, b=32),
            xaxis=dict(**_dark_axes(), title=dict(text="tiempo (s)", font=dict(size=10))),
            yaxis=dict(**_dark_axes(), title=dict(text="amplitud", font=dict(size=10))),
        )
        return fig

    # Sort all predictions; NO tight time pre-filter — per-beat loop clips to window.
    # A tight pre-filter based on start_offset silently drops all beats when the
    # .npy fragment's time reference doesn't match annotation time_second values.
    pdf = (
        predictions_df
        .sort_values(time_col)
        .reset_index(drop=True)
        .copy()
    )

    fig = go.Figure()

    # ── Detect binary vs multiclass ────────────────────────────────────────
    # Use the full (unfiltered) pdf for detection so an empty window doesn't
    # accidentally trigger multiclass mode.
    use_binary = (
        pred_col in pdf.columns
        and len(pdf) > 0
        and _is_binary_pred_col(pdf[pred_col])
    )

    # ══════════════════════════════════════════════════════════════════════
    # BINARY MODE — gray ECG + red vrect background for Anormal regions
    # ══════════════════════════════════════════════════════════════════════
    if use_binary:
        visible_start = start_offset
        visible_end   = t_end

        # Single gray ECG trace — never segmented
        fig.add_trace(go.Scatter(
            x=t_arr, y=valid_sig,
            mode="lines",
            name="ECG",
            line=dict(color="#cbd5e1", width=1.2),
            hovertemplate="t=%{x:.3f} s  amp=%{y:.4f}<extra></extra>",
            showlegend=False,
        ))

        # Red vrect per Anormal interval
        pdf_vis = pdf[
            (pdf[time_col] >= visible_start - 5) &
            (pdf[time_col] <= visible_end + 5)
        ].copy().reset_index(drop=True)

        vrect_count = 0

        if not pdf_vis.empty and pred_col in pdf_vis.columns:
            for i in range(len(pdf_vis)):
                row   = pdf_vis.iloc[i]
                seg_s = float(row[time_col])
                seg_e = (
                    float(pdf_vis.iloc[i + 1][time_col])
                    if i < len(pdf_vis) - 1
                    else visible_end
                )
                seg_s = max(seg_s, visible_start)
                seg_e = min(seg_e, visible_end)

                if seg_e <= seg_s:
                    continue

                if _is_abnormal_pred(row[pred_col]):
                    fig.add_vrect(
                        x0=seg_s, x1=seg_e,
                        fillcolor="rgba(239, 68, 68, 0.18)",
                        line_width=0,
                        layer="below",
                    )
                    vrect_count += 1

            # Fallback: loop produced no vrects but all visible preds are Anormal
            if vrect_count == 0:
                all_abn = all(
                    _is_abnormal_pred(r[pred_col]) for _, r in pdf_vis.iterrows()
                )
                if all_abn:
                    fig.add_vrect(
                        x0=visible_start, x1=visible_end,
                        fillcolor="rgba(239, 68, 68, 0.16)",
                        line_width=0, layer="below",
                    )
                    vrect_count += 1

        # Legend entry
        if vrect_count > 0:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(239, 68, 68, 0.6)", symbol="square"),
                name="Predicción: Anormal",
                showlegend=True,
            ))

        # Error markers: binarized real_col ≠ binarized pred_col
        if real_col in pdf.columns:
            pdf["_real_bin"] = pdf[real_col].apply(_rhythm_label_to_binary)
            pdf["_pred_bin"] = pdf[pred_col].apply(
                lambda x: "abnormal" if _is_abnormal_pred(x) else "normal"
            )
            err_df = pdf.loc[
                (pdf["_real_bin"] != pdf["_pred_bin"]) &
                (pdf[time_col] >= visible_start) &
                (pdf[time_col] <= visible_end)
            ]
            if not err_df.empty:
                has_btype = beat_type_col in err_df.columns
                err_hover = [
                    f"t = {row[time_col]:.3f} s<br>"
                    f"real: <b>{row[real_col]}</b> → binario: <b>{row['_real_bin']}</b><br>"
                    f"predicho: <b>{row['_pred_bin']}</b><br>"
                    + (f"beat_type: {row[beat_type_col]}" if has_btype else "")
                    for _, row in err_df.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=err_df[time_col].tolist(),
                    y=[marker_y] * len(err_df),
                    mode="markers",
                    marker=dict(symbol="x", size=9,
                                color="#f87171",
                                line=dict(color="#b91c1c", width=1.5)),
                    name="Error (real ≠ pred)",
                    showlegend=True,
                    text=err_hover,
                    hovertemplate="%{text}<extra></extra>",
                ))

    # ══════════════════════════════════════════════════════════════════════
    # MULTICLASS MODE — background regions + flat gray ECG line
    # ══════════════════════════════════════════════════════════════════════
    else:
        if len(pdf) > 0 and pred_col in pdf.columns:
            beat_times  = pdf[time_col].tolist()
            pred_labels = pdf[pred_col].tolist()

            if len(beat_times) >= 2:
                rr_last    = beat_times[-1] - beat_times[-2]
                next_times = beat_times[1:] + [beat_times[-1] + rr_last]
            else:
                next_times = [beat_times[0] + 1.0]

            # Group consecutive same-label abnormal runs → background shapes
            runs: list[tuple[float, float, str]] = []
            run_start = beat_times[0]
            run_label = pred_labels[0]
            for i in range(1, len(beat_times)):
                if pred_labels[i] != run_label:
                    if run_label != _NORMAL_LABEL:
                        runs.append((run_start, next_times[i - 1], run_label))
                    run_start = beat_times[i]
                    run_label = pred_labels[i]
            if run_label != _NORMAL_LABEL:
                runs.append((run_start, next_times[-1], run_label))

            for r_start, r_end, r_label in runs:
                rs = max(r_start, start_offset)
                re = min(r_end, t_end)
                if re <= rs:
                    continue
                fig.add_shape(
                    type="rect",
                    x0=rs, x1=re, y0=0, y1=1, yref="paper",
                    fillcolor=_REGION_COLORS.get(r_label, _REGION_DEFAULT),
                    line=dict(color=_REGION_BORDER.get(r_label, _REGION_DEFAULT_BORDER), width=1),
                    layer="below",
                )

        # ECG line (flat gray)
        fig.add_trace(go.Scatter(
            x=t_arr, y=valid_sig,
            mode="lines", name="ECG",
            line=dict(color="#d1d5db", width=1.1),
            hovertemplate="t=%{x:.3f} s  amp=%{y:.4f}<extra></extra>",
            showlegend=False,
        ))

        # Error markers (multiclass: direct label comparison)
        if len(pdf) > 0 and real_col in pdf.columns and pred_col in pdf.columns:
            err_df = pdf.loc[
                (pdf[real_col] != pdf[pred_col]) &
                (pdf[time_col] >= start_offset) &
                (pdf[time_col] <= t_end)
            ].copy()
            if not err_df.empty:
                has_btype = beat_type_col in err_df.columns
                err_hover = [
                    f"t = {row[time_col]:.3f} s<br>"
                    f"real: <b>{row[real_col]}</b><br>"
                    f"predicho: <b>{row[pred_col]}</b><br>"
                    + (f"beat_type: {row[beat_type_col]}" if has_btype else "")
                    for _, row in err_df.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=err_df[time_col].tolist(),
                    y=[marker_y] * len(err_df),
                    mode="markers",
                    marker=dict(symbol="x", size=9,
                                color="#f87171",
                                line=dict(color="#b91c1c", width=1.5)),
                    name="Error (real ≠ pred)",
                    showlegend=True,
                    text=err_hover,
                    hovertemplate="%{text}<extra></extra>",
                ))

        # Legend entries for anomalous rhythm labels present
        if len(pdf) > 0 and pred_col in pdf.columns:
            for lbl in sorted(
                lbl for lbl in pdf[pred_col].dropna().unique()
                if lbl != _NORMAL_LABEL
            ):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=12,
                                color=_RHYTHM_COLORS.get(lbl, "#5d6c8c"),
                                symbol="square", opacity=0.75),
                    name=lbl, showlegend=True,
                ))

    # ── Layout ────────────────────────────────────────────────────────────
    apply_dark_layout(
        fig,
        title=dict(text=title, font=dict(size=12, color="#c8d2e5")),
        height=height,
        margin=dict(l=44, r=10, t=40, b=32),
        xaxis=dict(**_dark_axes(), title=dict(text="tiempo (s)", font=dict(size=10))),
        yaxis=dict(**_dark_axes(), title=dict(text="amplitud", font=dict(size=10))),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# plot_ecg_with_binary_prediction_bands
# ---------------------------------------------------------------------------
def plot_ecg_with_binary_prediction_bands(
    signal,
    fs: float,
    pred_df: pd.DataFrame,
    time_col: str = "time_second",
    pred_col: str = "prediccion",
    segment_start_time: float = 0.0,
    max_seconds: float = 30.0,
    title: str = "ECG con regiones predichas por el modelo",
) -> go.Figure:
    """ECG gris con franjas rojas donde el modelo predice Anormal.

    El eje x del gráfico es RELATIVO (0 … max_seconds).
    Las predicciones se convierten a tiempo relativo restando segment_start_time,
    lo que resuelve el desajuste cuando time_second es absoluto (p. ej. 2000 s)
    y la señal mostrada empieza desde 0.
    """
    def _is_anormal(x) -> bool:
        s = str(x).strip().lower()
        return s in {"anormal", "abnormal", "no n", "non-normal", "not normal", "1", "true"}

    signal = np.asarray(signal).astype(float)
    n = min(len(signal), int(max_seconds * fs))
    signal = signal[:n]
    t_rel = np.arange(n) / fs  # eje x siempre relativo: 0 … max_seconds

    fig = go.Figure()

    # ── Rectángulos rojos (tiempo relativo) ──────────────────────────────────
    vrect_count = 0
    try:
        dfp = pred_df.copy()
        if time_col in dfp.columns and pred_col in dfp.columns:
            dfp["t_rel"] = pd.to_numeric(dfp[time_col], errors="coerce") - float(segment_start_time)
            dfp = dfp.dropna(subset=["t_rel"])
            dfp = dfp[(dfp["t_rel"] >= 0) & (dfp["t_rel"] <= max_seconds)].copy()
            dfp = dfp.sort_values("t_rel").reset_index(drop=True)

            for i in range(len(dfp)):
                pred  = dfp.loc[i, pred_col]
                start = float(dfp.loc[i, "t_rel"])
                end   = float(dfp.loc[i + 1, "t_rel"]) if i < len(dfp) - 1 else float(max_seconds)
                start = max(0.0, min(float(max_seconds), start))
                end   = max(0.0, min(float(max_seconds), end))

                if end > start and _is_anormal(pred):
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor="rgba(239, 68, 68, 0.22)",
                        line_width=0,
                        layer="below",
                    )
                    vrect_count += 1
    except Exception:
        pass

    # ── Línea ECG gris ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_rel,
        y=signal,
        mode="lines",
        line=dict(color="#cbd5e1", width=1.2),
        name="ECG",
        hovertemplate="tiempo=%{x:.2f}s<br>amplitud=%{y:.3f}<extra></extra>",
    ))

    # ── Leyenda dummy para zonas rojas ───────────────────────────────────────
    if vrect_count > 0:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color="rgba(239, 68, 68, 0.7)", symbol="square"),
            name="Predicción: Anormal",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#c8d2e5")),
        height=430,
        plot_bgcolor="#111827",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis=dict(
            title="tiempo relativo (s)",
            range=[0, max_seconds],
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title="amplitud",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
