"""Página 04 — Rendimiento del modelo.

Dos secciones separadas:
1. Modelo final oficial: Linear SVC, dataset completo, métricas definitivas.
2. Benchmark exploratorio de candidatos: 5 modelos, 150 casos — solo para selección.
"""

import json

import pandas as pd
import streamlit as st

from components.badges import badge
from components.cards  import callout, card_header, kv_table, metric_card, section_title
from components.charts import fit_time_bar, model_metrics_bar
from components.layout import page_header, page_footer
from utils.loaders     import (
    load_model_final_official,
    load_model_metadata,
    load_model_comparison_history,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
meta      = load_model_metadata()
df_final  = load_model_final_official()
df_hist   = load_model_comparison_history()

page_header(
    "Rendimiento del modelo",
    "Modelo final oficial (Linear SVC · dataset completo) y benchmark exploratorio "
    "de candidatos para detección binaria de ritmos intraoperatorios.",
    badge_html=badge("Evaluación final", "info"),
)

# ── Early exit ────────────────────────────────────────────────────────────────
if df_final is None or df_final.empty:
    callout(
        "err",
        "Datos no disponibles",
        "No se encontró <code>reports/tables/tabular_model_final_official.csv</code>. "
        "Ejecuta el pipeline para generarlo.",
    )
    st.stop()

# ── Parse final model row ─────────────────────────────────────────────────────
_row        = df_final.iloc[0]
winner_id   = str(_row.get("model", "linear_svc"))
winner_nice = winner_id.replace("_", " ").title()

def _get(col: str, fmt: str | None = None, fallback: str = "—") -> str:
    if col not in _row.index:
        return fallback
    v = _row[col]
    if pd.isna(v):
        return fallback
    return format(float(v), fmt) if fmt else str(v)

def _fmt_model(name: str) -> str:
    return str(name).replace("_", " ").title()

def _fmt_param_value(v) -> str:
    try:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, int):
            return str(v)
    except Exception:
        pass
    return str(v)


def _fmt_params(raw) -> str:
    if pd.isna(raw) or raw == "":
        return "No disponible"
    try:
        params = json.loads(str(raw))
        cleaned = {k.replace("clf__", "").replace("__", "_"): v for k, v in params.items()}
        return " · ".join(f"{k} = {_fmt_param_value(v)}" for k, v in cleaned.items())
    except Exception:
        s = str(raw)
        return s[:120] + "…" if len(s) > 120 else s

_MODEL_CHIPS: dict[str, list[tuple[str, str]]] = {
    "linear_svc":    [("lineal", "info"), ("L2 hinge", "muted"), ("balanced", "ok")],
    "random_forest": [("ensemble", "info"), ("bagging", "muted"), ("balanced", "ok")],
    "xgboost":       [("boosting", "info"), ("tree-based", "muted")],
    "mlp":           [("neural net", "info"), ("relu", "muted"), ("adam", "muted")],
    "decision_tree": [("árbol", "info"), ("entropy", "muted")],
    "logreg":        [("lineal", "info"), ("L-BFGS", "muted"), ("balanced", "ok")],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Callout metodológico
# ═══════════════════════════════════════════════════════════════════════════════
callout(
    "info",
    "Metodología de evaluación — dos componentes",
    "Esta página presenta <b>dos resultados separados</b>: "
    "(1) el <b>modelo final oficial</b> — Linear SVC entrenado con el dataset completo "
    "(639 460 registros, 482 casos) — y "
    "(2) un <b>benchmark exploratorio</b> con 5 candidatos sobre 150 casos "
    "(<code>--max-cases 150 --n-iter 5 --n-splits 3</code>). "
    "El benchmark exploratorio <b>no representa métricas finales</b>; "
    "fue una corrida rápida de selección de candidatos. "
    "La tarea es <b>binaria</b>: normal (<code>rhythm_label == N</code>) "
    "vs anormal (<code>rhythm_label != N</code>). "
    "Métrica principal: <b>F1-macro</b> — promedia F1 de ambas clases con peso uniforme. "
    "Resultados <b>académicos</b> — no para uso clínico.",
)

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Modelo final oficial
# ═══════════════════════════════════════════════════════════════════════════════
section_title("Modelo final oficial")

col_win, col_kpis = st.columns([1.35, 1])

with col_win:
    with st.container(border=True):
        card_header(
            "Modelo desplegado",
            "corrida definitiva · 639 460 registros · 482 casos",
            right_html=(
                badge("winner", "winner") + "&nbsp;" +
                badge("full dataset", "ok") + "&nbsp;" +
                badge("binario", "info")
            ),
        )

        chips = _MODEL_CHIPS.get(winner_id, [])
        chips_html = " ".join(badge(t, k) for t, k in chips)
        st.markdown(
            f"""<div style="display:flex;align-items:baseline;gap:14px;margin:6px 0 10px">
                  <div style="font-size:28px;font-weight:600;color:var(--fg-0);
                              letter-spacing:-.01em">{winner_nice}</div>
                  <span style="font-family:var(--mono);font-size:11px;color:var(--fg-3)">
                    scikit-learn
                  </span>
                </div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">
                  {chips_html}
                </div>""",
            unsafe_allow_html=True,
        )

        kv_table([
            ("F1-macro (test)",
             f'<b style="color:var(--teal)">{_get("test_f1_macro", ".3f")}</b>'),
            ("Balanced accuracy",  _get("test_balanced_accuracy", ".3f")),
            ("Accuracy",           _get("test_accuracy",          ".3f")),
            ("Precision (macro)",  _get("test_precision_macro",   ".3f")),
            ("Recall (macro)",     _get("test_recall_macro",      ".3f")),
            ("Tiempo entreno",     f"{float(_row['fit_seconds']):.0f} s "
                                   f"({float(_row['fit_seconds'])/60:.1f} min)"
                                   if "fit_seconds" in _row.index and not pd.isna(_row["fit_seconds"])
                                   else "—"),
        ])

        # Hyperparameters from metadata
        if meta and "best_params" in meta:
            params_str = _fmt_params(meta["best_params"])
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            st.caption("Mejores hiperparámetros:")
            st.code(params_str, language=None)

with col_kpis:
    r1, r2 = st.columns(2)
    r3, r4 = st.columns(2)
    r5, r6 = st.columns(2)

    with r1:
        metric_card("Modelo", winner_nice, accent="teal")
    with r2:
        metric_card("F1-macro test", _get("test_f1_macro", ".3f"), accent="teal")
    with r3:
        metric_card("Balanced acc.", _get("test_balanced_accuracy", ".3f"), accent="blue")
    with r4:
        metric_card("Accuracy test", _get("test_accuracy", ".3f"), accent="blue")
    with r5:
        metric_card("Precision macro", _get("test_precision_macro", ".3f"), accent="warn")
    with r6:
        t_fit = (
            float(_row["fit_seconds"])
            if "fit_seconds" in _row.index and not pd.isna(_row["fit_seconds"])
            else None
        )
        metric_card("Tiempo entreno", f"{t_fit:.0f} s" if t_fit else "—", accent="warn")


st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Benchmark exploratorio de candidatos
# ═══════════════════════════════════════════════════════════════════════════════
section_title("Benchmark exploratorio de candidatos")

callout(
    "warn",
    "Corrida exploratoria — no es el resultado final",
    "Esta comparación se realizó con <b>150 casos</b> usando parámetros reducidos "
    "(<code>--max-cases 150 --n-iter 5 --n-splits 3</code>). "
    "Su único propósito fue <b>seleccionar candidatos</b> para la corrida definitiva, "
    "<b>no evaluar el rendimiento real del modelo</b>. "
    "MLP obtuvo el mayor F1-macro en esta corrida, pero "
    "<b>no fue seleccionado como modelo final</b> porque: "
    "(a) la corrida fue exploratoria con pocos datos, "
    "(b) presentó advertencias de convergencia, y "
    "(c) no ofrece interpretabilidad directa. "
    "El <b>modelo final oficial es Linear SVC</b> entrenado con el dataset completo.",
)

st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

if df_hist is not None:
    df_exp = (
        df_hist[df_hist["run_type"] == "exploratory_150_cases"].copy()
        if "run_type" in df_hist.columns else df_hist.copy()
    )
    if "test_f1_macro" in df_exp.columns:
        df_exp = df_exp.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

    exp_labels       = [_fmt_model(m) for m in df_exp["model"]] if not df_exp.empty else []
    exp_winner_label = "Mlp"

    col_h1, col_h2 = st.columns([1.5, 1])

    with col_h1:
        with st.container(border=True):
            card_header(
                "Métricas por candidato",
                "corrida exploratoria · 150 casos · F1 · Precisión · Recall · Accuracy",
            )
            metrics_exp: dict[str, list[float]] = {}
            for col_key, display_name in [
                ("test_f1_macro",        "F1-macro"),
                ("test_precision_macro", "Precision"),
                ("test_recall_macro",    "Recall"),
                ("test_accuracy",        "Accuracy"),
            ]:
                if col_key in df_exp.columns and not df_exp.empty:
                    metrics_exp[display_name] = df_exp[col_key].tolist()

            if metrics_exp and exp_labels:
                fig_exp = model_metrics_bar(
                    labels=exp_labels,
                    metrics=metrics_exp,
                    winner_label=exp_winner_label,
                    height=310,
                )
                st.plotly_chart(fig_exp, use_container_width=True, config={"displayModeBar": False})
                st.caption(
                    "MLP destacado como mejor en esta corrida exploratoria (150 casos). "
                    "No se seleccionó como modelo final — ver nota arriba. "
                    "Modelo desplegado: Linear SVC (corrida con dataset completo)."
                )
            else:
                st.warning("No hay métricas suficientes para graficar.")

    with col_h2:
        with st.container(border=True):
            card_header("Tiempo de entrenamiento", "corrida exploratoria · segundos")

            if "fit_seconds" in df_exp.columns and not df_exp.empty:
                fig_time_exp = fit_time_bar(
                    labels=exp_labels,
                    times=df_exp["fit_seconds"].tolist(),
                    winner_label=exp_winner_label,
                    height=310,
                )
                st.plotly_chart(
                    fig_time_exp, use_container_width=True, config={"displayModeBar": False}
                )
                callout(
                    "info",
                    "Trade-off velocidad / calidad",
                    "MLP tiene el mayor F1 exploratorio pero también el mayor tiempo y no "
                    "converge completamente. Linear SVC ofrece buen balance entre "
                    "rendimiento, velocidad e interpretabilidad, decisivo para la "
                    "corrida con el dataset completo.",
                )
            else:
                st.info("Tiempo de entrenamiento no disponible.")

    # ── Tabla histórica ──────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        card_header(
            "Todos los candidatos — benchmark exploratorio",
            "150 casos · --n-iter 5 · --n-splits 3 · no son métricas finales",
            right_html=badge(f"{len(df_exp)} modelos", "muted"),
        )

        display_cols = [c for c in [
            "model", "status",
            "test_f1_macro", "test_precision_macro", "test_recall_macro",
            "test_accuracy", "test_balanced_accuracy", "fit_seconds", "notes",
        ] if c in df_exp.columns]

        df_display = df_exp[display_cols].copy()
        if "model" in df_display.columns:
            df_display["model"] = df_display["model"].apply(_fmt_model)

        col_rename = {
            "model":                  "Modelo",
            "status":                 "Estado",
            "test_f1_macro":          "F1-macro",
            "test_precision_macro":   "Precisión",
            "test_recall_macro":      "Recall",
            "test_accuracy":          "Accuracy",
            "test_balanced_accuracy": "Balanced Acc.",
            "fit_seconds":            "Tiempo (s)",
            "notes":                  "Nota",
        }
        df_display = df_display.rename(columns=col_rename)

        fmt_cols = {k: "{:.3f}" for k in
                    ["F1-macro", "Precisión", "Recall", "Accuracy", "Balanced Acc."]
                    if k in df_display.columns}
        if "Tiempo (s)" in df_display.columns:
            fmt_cols["Tiempo (s)"] = "{:.1f}"

        st.dataframe(
            df_display.style.format(fmt_cols, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Corrida exploratoria con 150 casos. "
            "MLP ganó en F1 pero no fue seleccionado — ver nota de la sección. "
            "El modelo oficial es Linear SVC entrenado con el dataset completo."
        )

else:
    callout(
        "warn",
        "Benchmark exploratorio no disponible",
        "No se encontró <code>reports/tables/tabular_model_comparison_history.csv</code>.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Estrategia de búsqueda (expander)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Estrategia de búsqueda — modelo final", expanded=False):
    if meta:
        card_header("Corrida final · metodología", "dataset completo")
        kv_table([
            ("Método",        "RandomizedSearchCV"),
            ("Split",         "StratifiedGroupKFold · por case_id"),
            ("Scoring",       "f1_macro"),
            ("Train groups",  str(meta.get("n_train_groups", "—"))),
            ("Test groups",   str(meta.get("n_test_groups",  "—"))),
            ("Features orig.","73 (57 num + 16 cat)"),
            ("Features OHE",  "162 tras One-Hot Encoding"),
            ("Target",        "binario: normal (N) vs anormal (no N)"),
            ("Fecha entreno", str(meta.get("training_datetime", meta.get("trained_at", "—")))[:19]),
        ])

        if "best_params" in meta:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            st.caption("Mejores hiperparámetros (modelo final):")
            st.code(_fmt_params(meta["best_params"]), language=None)
    else:
        st.info("Metadata no disponible.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Interpretación
# ═══════════════════════════════════════════════════════════════════════════════
section_title("Interpretación de resultados")

with st.container(border=True):
    card_header("Conclusión de esta sección", "lectura cualitativa")
    st.markdown(
        f"""<div style="font-size:13.5px;color:var(--fg-1);line-height:1.75">
              <p>
                El modelo final <b style="color:var(--teal)">Linear SVC</b> entrenado con el
                dataset completo (639 460 registros, 482 casos) obtuvo
                <b>F1-macro = {_get("test_f1_macro", ".3f")}</b>,
                <b>balanced accuracy = {_get("test_balanced_accuracy", ".3f")}</b> y
                <b>accuracy = {_get("test_accuracy", ".3f")}</b> en el conjunto de test.
                Este es el <b>modelo oficial del proyecto</b>.
              </p>
              <p>
                El benchmark exploratorio (150 casos) sirvió para comparar candidatos.
                MLP obtuvo el mayor F1-macro en esa corrida rápida (0.718), pero
                <b>no fue seleccionado</b> como modelo final por presentar advertencias
                de convergencia, no ofrecer interpretabilidad directa y porque sus
                métricas se obtuvieron con solo 150 casos — no con el dataset completo.
              </p>
              <p>
                La tarea es <b>binaria</b>: cada latido se clasifica como
                <b>normal</b> (<code>rhythm_label == "N"</code>, ritmo sinusal) o
                <b>anormal</b> (<code>rhythm_label != "N"</code>, cualquier arritmia).
                Las etiquetas originales de ritmo de la VitalDB se usan únicamente para
                construir esta variable binaria; el modelo no distingue subtipos de arritmia.
              </p>
              <p>
                El <b>F1-macro ({_get("test_f1_macro", ".3f")})</b> es la métrica principal
                porque evalúa ambas clases con peso uniforme. La accuracy
                ({_get("test_accuracy", ".3f")}) puede ser engañosamente alta dado el
                desbalance (NSR domina ~65% de los latidos).
                Los detalles por clase se exploran en <b>Evaluación por clase</b>.
              </p>
            </div>""",
        unsafe_allow_html=True,
    )

    callout(
        "warn",
        "Limitación metodológica",
        "Estos resultados corresponden a un pipeline académico. "
        "Los valores de F1-macro podrían mejorar con más iteraciones de búsqueda, "
        "re-balanceo de clases (SMOTE) o features RR más granulares.",
    )

page_footer()
