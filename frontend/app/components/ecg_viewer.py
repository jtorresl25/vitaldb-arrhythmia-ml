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
    """ECG trace con regiones coloreadas solo donde el modelo predice ritmo anómalo.

    Reglas visuales:
    - ``pred == "N"``:     sin sombreado (fondo oscuro limpio).
    - ``pred != "N"``:     franja de color sobre la señal ECG.
    - ``real != pred``:    marcador rojo ✕ en la cima de la señal, con tooltip detallado.

    La señal ECG aparece en gris claro encima de los sombreados.
    """
    # --- Señal recortada a la ventana de visualización ---
    sig = np.asarray(signal, dtype=float).ravel()
    n_disp = min(len(sig), int(max_seconds * fs))
    t_arr = _time_axis(n_disp, fs, offset=start_offset)
    t_end = float(t_arr[-1])

    # Amplitud real de la señal para posicionar marcadores
    valid_sig = sig[:n_disp]
    sig_valid_vals = valid_sig[~np.isnan(valid_sig)]
    y_max = float(np.max(sig_valid_vals)) if len(sig_valid_vals) else 1.0
    y_min = float(np.min(sig_valid_vals)) if len(sig_valid_vals) else -1.0
    y_span = y_max - y_min if y_max > y_min else 1.0
    # Marcadores de error: 90% de la altura máxima de la señal
    marker_y = y_max + y_span * 0.08

    # --- Filtrar latidos en la ventana ---
    pdf = predictions_df.copy()
    pdf = pdf.loc[
        (pdf[time_col] >= start_offset - 1.0) &
        (pdf[time_col] <= t_end + 1.0)
    ].sort_values(time_col).reset_index(drop=True)

    fig = go.Figure()

    # --- Regiones de fondo: solo para predicciones anómalas ---
    if len(pdf) > 0 and pred_col in pdf.columns:
        beat_times  = pdf[time_col].tolist()
        pred_labels = pdf[pred_col].tolist()

        # Calcular límite derecho de cada región = tiempo del latido siguiente
        if len(beat_times) >= 2:
            rr_last = beat_times[-1] - beat_times[-2]
            next_times = beat_times[1:] + [beat_times[-1] + rr_last]
        else:
            next_times = [beat_times[0] + 1.0]

        # Agrupar runs consecutivos con el mismo label anómalo
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
            fill   = _REGION_COLORS.get(r_label, _REGION_DEFAULT)
            border = _REGION_BORDER.get(r_label, _REGION_DEFAULT_BORDER)
            fig.add_shape(
                type="rect",
                x0=rs, x1=re,
                y0=0, y1=1, yref="paper",
                fillcolor=fill,
                line=dict(color=border, width=1),
                layer="below",
            )

    # --- Traza ECG ---
    fig.add_trace(go.Scatter(
        x=t_arr,
        y=valid_sig,
        mode="lines",
        name="ECG",
        line=dict(color="#d1d5db", width=1.1),
        hovertemplate="t=%{x:.3f} s  amp=%{y:.4f}<extra></extra>",
        showlegend=False,
    ))

    # --- Marcadores de error: real ≠ pred ---
    if len(pdf) > 0 and real_col in pdf.columns and pred_col in pdf.columns:
        err_df = pdf.loc[
            (pdf[real_col] != pdf[pred_col]) &
            (pdf[time_col] >= start_offset) &
            (pdf[time_col] <= t_end)
        ].copy()
        if not err_df.empty:
            has_btype = beat_type_col in err_df.columns

            err_hover = []
            for _, row in err_df.iterrows():
                bt = str(row[beat_type_col]) if has_btype else "—"
                txt = (
                    f"t = {row[time_col]:.3f} s<br>"
                    f"real: <b>{row[real_col]}</b><br>"
                    f"predicho: <b>{row[pred_col]}</b><br>"
                    f"beat_type: {bt}<br>"
                    f"correcto: False"
                )
                err_hover.append(txt)

            fig.add_trace(go.Scatter(
                x=err_df[time_col].tolist(),
                y=[marker_y] * len(err_df),
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=9,
                    color="#f87171",
                    line=dict(color="#b91c1c", width=1.5),
                ),
                name="Error (real ≠ pred)",
                showlegend=True,
                text=err_hover,
                hovertemplate="%{text}<extra></extra>",
            ))

    # --- Entradas de leyenda para ritmos anómalos presentes ---
    if len(pdf) > 0 and pred_col in pdf.columns:
        anomalous = sorted(
            lbl for lbl in pdf[pred_col].dropna().unique()
            if lbl != _NORMAL_LABEL
        )
        for lbl in anomalous:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    size=12,
                    color=_RHYTHM_COLORS.get(lbl, "#5d6c8c"),
                    symbol="square",
                    opacity=0.75,
                ),
                name=lbl,
                showlegend=True,
            ))

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
