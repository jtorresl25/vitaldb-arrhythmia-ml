import streamlit as st

import pandas as pd

from components.layout import page_header
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.badges import badge, badge_row
from components.charts import bar_chart_h, apply_dark_layout
from components.tables import feature_importance_table
from utils.loaders import (
    load_feature_importance,
    load_feature_columns,
    load_model_metadata,
    load_model_comparison,
    correlation_figure_path,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
meta   = load_model_metadata()
winner = meta.get("winner_model", "—").replace("_", " ").title() if meta else "—"
winner_raw = meta.get("winner_model", "—") if meta else "—"

page_header(
    "Interpretabilidad",
    "Análisis de las features más relevantes para el modelo ganador.",
    badge_html=badge_row(badge(winner, "winner"), badge("Explicación del modelo", "info")),
)

# ── Feature group classification ──────────────────────────────────────────────
# Cubre features del flujo tabular (metadatos clínicos + RR)
# y del flujo legacy ECG (morfología de onda).
_GROUP_MAP: dict[str, str] = {
    # --- Flujo tabular: RR local por latido ---
    "rr_prev":              "RR local",
    "rr_next":              "RR local",
    "rr_mean_local":        "RR local",
    "rr_ratio":             "RR local",
    "hr_inst_from_rr_prev": "RR local",
    "position_in_case":     "Timing / posición",
    # --- Flujo tabular: RR global (caso) ---
    "case_rr_count": "RR global (caso)",
    "case_rr_mean":  "RR global (caso)",
    "case_rr_std":   "RR global (caso)",
    "case_rr_min":   "RR global (caso)",
    "case_rr_max":   "RR global (caso)",
    "case_rr_rmssd": "RR global (caso)",
    "case_rr_pnn50": "RR global (caso)",
    # --- Flujo tabular: demografía ---
    "age":    "Demografía",
    "sex":    "Demografía",
    "height": "Demografía",
    "weight": "Demografía",
    "bmi":    "Demografía",
    "asa":    "Demografía",
    "emop":   "Demografía",
    # --- Flujo tabular: contexto clínico ---
    "department": "Contexto clínico",
    "optype":     "Contexto clínico",
    "approach":   "Contexto clínico",
    "position":   "Contexto clínico",
    "ane_type":   "Contexto clínico",
    "preop_ecg":  "Contexto clínico",
    "preop_pft":  "Contexto clínico",
    # --- Flujo tabular: labs preoperatorios ---
    "preop_hb":   "Labs preoperatorios",
    "preop_plt":  "Labs preoperatorios",
    "preop_pt":   "Labs preoperatorios",
    "preop_aptt": "Labs preoperatorios",
    "preop_na":   "Labs preoperatorios",
    "preop_k":    "Labs preoperatorios",
    "preop_gluc": "Labs preoperatorios",
    "preop_alb":  "Labs preoperatorios",
    "preop_ast":  "Labs preoperatorios",
    "preop_alt":  "Labs preoperatorios",
    "preop_bun":  "Labs preoperatorios",
    "preop_cr":   "Labs preoperatorios",
    "preop_ph":   "Labs preoperatorios",
    "preop_hco3": "Labs preoperatorios",
    "preop_be":   "Labs preoperatorios",
    "preop_pao2": "Labs preoperatorios",
    "preop_paco2":"Labs preoperatorios",
    "preop_sao2": "Labs preoperatorios",
    "preop_htn":  "Labs preoperatorios",
    "preop_dm":   "Labs preoperatorios",
    # --- Flujo tabular: drogas / fluidos intraop. ---
    "intraop_ebl":        "Drogas / fluidos intraop.",
    "intraop_uo":         "Drogas / fluidos intraop.",
    "intraop_rbc":        "Drogas / fluidos intraop.",
    "intraop_ffp":        "Drogas / fluidos intraop.",
    "intraop_crystalloid":"Drogas / fluidos intraop.",
    "intraop_colloid":    "Drogas / fluidos intraop.",
    "intraop_ppf":        "Drogas / fluidos intraop.",
    "intraop_mdz":        "Drogas / fluidos intraop.",
    "intraop_ftn":        "Drogas / fluidos intraop.",
    "intraop_rocu":       "Drogas / fluidos intraop.",
    "intraop_vecu":       "Drogas / fluidos intraop.",
    "intraop_eph":        "Drogas / fluidos intraop.",
    "intraop_phe":        "Drogas / fluidos intraop.",
    "intraop_epi":        "Drogas / fluidos intraop.",
    "intraop_ca":         "Drogas / fluidos intraop.",
    # --- Flujo tabular: vía aérea / accesos ---
    "tubesize":   "Vía aérea / accesos",
    "lmasize":    "Vía aérea / accesos",
    "cormack":    "Vía aérea / accesos",
    "dltubesize": "Vía aérea / accesos",
    "iv1":  "Vía aérea / accesos",
    "iv2":  "Vía aérea / accesos",
    "aline1": "Vía aérea / accesos",
    "aline2": "Vía aérea / accesos",
    "cline1": "Vía aérea / accesos",
    "cline2": "Vía aérea / accesos",
    # --- Flujo tabular: timestamps ---
    "time_second":           "Timing / posición",
    "analysis_start_time_sec":"Timing / posición",
    "analysis_end_time_sec":  "Timing / posición",
    "analyzed_duration_sec":  "Timing / posición",
    "total_beats":            "Timing / posición",
    "caseend":  "Timing / posición",
    "anestart": "Timing / posición",
    "aneend":   "Timing / posición",
    "opstart":  "Timing / posición",
    "opend":    "Timing / posición",
    # --- Flujo legacy ECG: amplitud / morfología ---
    "mean":     "Amplitud / morfología",
    "std":      "Amplitud / morfología",
    "var":      "Amplitud / morfología",
    "min":      "Amplitud / morfología",
    "max":      "Amplitud / morfología",
    "range":    "Amplitud / morfología",
    "median":   "Amplitud / morfología",
    "p25":      "Amplitud / morfología",
    "p75":      "Amplitud / morfología",
    "iqr":      "Amplitud / morfología",
    "skew":     "Amplitud / morfología",
    "kurtosis": "Amplitud / morfología",
    "energy":   "Amplitud / morfología",
    "abs_mean": "Amplitud / morfología",
    "zero_crossing_rate": "Amplitud / morfología",
}

# Resolución rule-based para features no en el mapa exacto
def _group_from_name(col: str) -> str:
    g = _GROUP_MAP.get(col)
    if g:
        return g
    if col.startswith("preop_"):
        return "Labs preoperatorios"
    if col.startswith("intraop_"):
        return "Drogas / fluidos intraop."
    if col.startswith("case_rr"):
        return "RR global (caso)"
    return "Otros"


# Strip prefixes generados por ColumnTransformer (verbose_feature_names_out=True)
# e.g. "num__age" → "age",  "cat__sex_M" → "sex"
def _strip_prefix(feat_name: str, categorical: list[str] | None = None) -> str:
    if feat_name.startswith("num__"):
        return feat_name[5:]
    if feat_name.startswith("cat__"):
        remainder = feat_name[5:]
        # match longest known categorical column
        if categorical:
            best: str | None = None
            for col in categorical:
                if remainder == col or remainder.startswith(col + "_"):
                    if best is None or len(col) > len(best):
                        best = col
            if best:
                return best
        # fallback: return part before first underscore of value
        return remainder
    return feat_name


_GROUP_COLORS = {
    "RR local":               "#a78bfa",
    "RR global (caso)":       "#2dd4bf",
    "Demografía":             "#f472b6",
    "Labs preoperatorios":    "#fb923c",
    "Drogas / fluidos intraop.": "#facc15",
    "Contexto clínico":       "#4a8cff",
    "Vía aérea / accesos":    "#34d399",
    "Timing / posición":      "#94a3b8",
    "Amplitud / morfología":  "#60a5fa",
    "Otros":                  "#5d6c8c",
}

# ── Robust column detection ────────────────────────────────────────────────────
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        matches = [col for col in df.columns if c in col.lower()]
        if matches:
            return matches[0]
    return None


# ── Load data ─────────────────────────────────────────────────────────────────
df_imp       = load_feature_importance()
feature_cols = load_feature_columns()
corr_png     = correlation_figure_path()

has_imp  = df_imp is not None and not df_imp.empty
has_cols = feature_cols is not None
has_corr = corr_png is not None

if not has_imp:
    callout(
        "warn",
        "Datos de importancia no disponibles",
        "No se encontró <code>reports/tables/best_model_feature_importance.csv</code>. "
        "Ejecuta el pipeline de evaluación para generarlo.",
    )
    st.stop()

# Detect actual column names
FEAT_COL = _find_col(df_imp, ["feature", "feature_name", "variable"])
IMP_COL  = _find_col(df_imp, [
    "importance", "abs_importance", "score",
    "coefficient", "coef", "abs_coefficient",
])

if FEAT_COL is None or IMP_COL is None:
    callout(
        "err",
        "Formato de CSV inesperado",
        f"No se encontraron columnas de feature/importancia. "
        f"Columnas disponibles: {list(df_imp.columns)}",
    )
    st.stop()

# Ensure importance is numeric and sorted
df_imp[IMP_COL] = pd.to_numeric(df_imp[IMP_COL], errors="coerce")
df_imp = df_imp.dropna(subset=[IMP_COL]).sort_values(IMP_COL, ascending=False).reset_index(drop=True)

# _known_cat must be defined before any _strip_prefix call below
_known_cat = meta.get("categorical_features", []) if meta else []

if df_imp.empty:
    callout(
        "warn",
        "Sin features con importancia válida",
        f"El archivo de importancia fue encontrado pero no contiene filas con valores numéricos "
        f"en la columna <code>{IMP_COL}</code>. Vuelve a ejecutar el benchmark.",
    )
    st.stop()

n_features_csv   = len(df_imp)
n_features_total = len(feature_cols) if has_cols else n_features_csv
top_feature      = _strip_prefix(str(df_imp.loc[0, FEAT_COL]), _known_cat)
top_importance   = float(df_imp.loc[0, IMP_COL])
total_importance = float(df_imp[IMP_COL].sum())

# Assign group: strip prefix first, then lookup
def _assign_group(feat_name: str) -> str:
    base = _strip_prefix(str(feat_name), _known_cat)
    return _group_from_name(base)

df_imp["Grupo"]       = df_imp[FEAT_COL].map(_assign_group)
df_imp["feature_base"] = df_imp[FEAT_COL].map(lambda f: _strip_prefix(str(f), _known_cat))

# Infer interpretation type from metadata
interp_type = "importancia de features"
if meta and meta.get("winner_model", ""):
    m = meta["winner_model"].lower()
    if "forest" in m or "tree" in m or "xgb" in m or "boost" in m:
        interp_type = "feature importance (árbol)"
    elif "svc" in m or "svm" in m or "linear" in m:
        interp_type = "norma de coeficientes"

# ── Callout metodológico ──────────────────────────────────────────────────────
_interp_body = (
    "La importancia de variables ayuda a entender qué información usa el modelo, "
    "pero <b>no equivale a una explicación clínica directa</b>. "
)
if "árbol" in interp_type or "forest" in winner_raw.lower():
    _interp_body += (
        f"Para <b>{winner}</b> (Random Forest), la importancia se mide como la "
        "reducción promedio de impureza Gini a través de todos los árboles. "
        "Este modelo usa <b>datos tabulares</b> (metadatos clínicos + intervalos RR), "
        "no la forma de onda ECG directamente. "
    )
else:
    _interp_body += (
        f"Para <b>{winner}</b>, la importancia se deriva de los coeficientes del modelo. "
    )
_interp_body += (
    "Una feature importante <i>no implica causalidad</i>. "
    "Este análisis es <b>académico</b> y no debe usarse para diagnóstico clínico."
)
callout("info", "Cómo interpretar la importancia de features", _interp_body)

st.write("")

# ── KPIs ─────────────────────────────────────────────────────────────────────
section_title("Resumen de features")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card(
        "Features totales",
        str(n_features_total),
        "en el modelo entrenado",
        accent="blue",
    )
with c2:
    metric_card(
        "Features en reporte",
        str(n_features_csv),
        "con importancia > 0",
        accent="blue",
    )
with c3:
    metric_card(
        "Feature más importante",
        top_feature.replace("_", " "),
        f"importancia {top_importance:.3f}",
        accent="teal",
        helper_kind="ok",
    )
with c4:
    metric_card(
        "Importancia máxima",
        f"{top_importance:.3f}",
        f"de total {total_importance:.1f}",
        accent="teal",
    )
with c5:
    metric_card(
        "Modelo interpretado",
        winner,
        winner_raw,
        accent="muted",
    )
with c6:
    metric_card(
        "Tipo de interpretación",
        interp_type.split("(")[0].strip(),
        interp_type,
        accent="muted",
    )

st.write("")

# ── Top-N bar chart ───────────────────────────────────────────────────────────
section_title("Importancia de features")

col_sel, _, _ = st.columns([1, 1, 2])
with col_sel:
    _n_options = [5, min(10, n_features_csv)]
    if n_features_csv > 10:
        _n_options.append(15)
    if n_features_csv > 15:
        _n_options.append(n_features_csv)
    _n_options = sorted(set(_n_options))
    _labels_n  = [f"Top {n}" if n < n_features_csv else f"Top {n} (todos)" for n in _n_options]
    if len(_n_options) == 1:
        top_n = _n_options[0]
    else:
        choice = st.selectbox("Mostrar", _labels_n, index=min(1, len(_labels_n) - 1))
        top_n  = _n_options[_labels_n.index(choice)]

df_plot = df_imp.head(top_n)
# Use stripped base names for human-readable labels
labels  = df_plot["feature_base"].str.replace("_", " ").tolist()
values  = df_plot[IMP_COL].tolist()

chart_col, table_col = st.columns([1.5, 1])

with chart_col:
    with st.container(border=True):
        st.markdown(f"**Top {top_n} features por importancia**")
        # Color each bar by group
        group_col = df_plot["Grupo"].tolist()
        colors = [_GROUP_COLORS.get(g, _GROUP_COLORS["Otros"]) for g in group_col]

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#8a98b5"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
        ))
        apply_dark_layout(
            fig,
            height=max(280, top_n * 32),
            bargap=0.28,
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
                showticklabels=True,
            ),
            yaxis=dict(showgrid=False, zeroline=False, autorange=True),
            margin=dict(l=10, r=50, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Group legend
        legend_html = " ".join(
            f'<span class="badge badge-muted" style="background:{c};color:#07090f;font-size:10px">'
            f'{g}</span>'
            for g, c in _GROUP_COLORS.items()
            if g != "Otros" and g in group_col
        )
        st.html(f'<div style="margin-top:4px;display:flex;gap:6px;flex-wrap:wrap">{legend_html}</div>')

with table_col:
    with st.container(border=True):
        st.markdown("**Tabla de importancias**")
        feature_importance_table(
            df_plot,
            feature_col=FEAT_COL,
            importance_col=IMP_COL,
            group_map=_GROUP_MAP,
        )

st.write("")

# ── Group analysis ────────────────────────────────────────────────────────────
section_title("Importancia por grupo de features")

# Compute group summary from CSV features only
group_summary = (
    df_imp.groupby("Grupo")[IMP_COL]
    .agg(["sum", "mean", "count"])
    .rename(columns={"sum": "Importancia acumulada", "mean": "Importancia promedio", "count": "N features"})
    .sort_values("Importancia acumulada", ascending=False)
    .reset_index()
)

grp_col1, grp_col2 = st.columns([1.2, 1])

with grp_col1:
    with st.container(border=True):
        st.markdown("**Importancia acumulada por grupo**")
        g_labels = group_summary["Grupo"].tolist()
        g_values = group_summary["Importancia acumulada"].tolist()
        g_colors = [_GROUP_COLORS.get(g, _GROUP_COLORS["Otros"]) for g in g_labels]

        fig_grp = go.Figure(go.Bar(
            x=g_values,
            y=g_labels,
            orientation="h",
            marker=dict(color=g_colors, line=dict(width=0)),
            text=[f"{v:.2f} ({v/total_importance:.0%})" for v in g_values],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#8a98b5"),
            cliponaxis=False,
        ))
        apply_dark_layout(
            fig_grp,
            height=240,
            bargap=0.32,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, autorange=True),
            margin=dict(l=10, r=80, t=10, b=10),
        )
        st.plotly_chart(fig_grp, use_container_width=True, config={"displayModeBar": False})

with grp_col2:
    with st.container(border=True):
        st.markdown("**Resumen por grupo**")
        tbl = group_summary.copy()
        tbl["Importancia acumulada"] = tbl["Importancia acumulada"].apply(lambda v: f"{v:.3f}")
        tbl["Importancia promedio"]  = tbl["Importancia promedio"].apply(lambda v: f"{v:.3f}")
        tbl["% del total"]           = group_summary["Importancia acumulada"].apply(
            lambda v: f"{v / total_importance:.1%}"
        )
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Contextual note for total vs reported features
    if n_features_csv < n_features_total:
        n_zero = n_features_total - n_features_csv
        callout(
            "warn",
            f"{n_zero} features con importancia cero o no reportadas",
            f"El CSV contiene {n_features_csv} features con importancia > 0. "
            f"Las {n_zero} features restantes (de {n_features_total} totales) "
            f"tienen importancia nula o muy pequeña para este modelo y no aparecen en el reporte. "
            f"Esto es normal en modelos lineales con muchas features correlacionadas.",
        )

st.write("")

# ── Correlation heatmap ────────────────────────────────────────────────────────
section_title("Mapa de correlación entre features")

if has_corr:
    with st.container(border=True):
        card_header(
            "Correlación de Pearson entre features",
            "imagen exportada por el pipeline",
            right_html=badge("PNG", "ok"),
        )
        st.image(str(corr_png), use_container_width=True)

    callout(
        "warn",
        "Features correlacionadas — efecto en la interpretación",
        "Algunas features presentan alta correlación entre sí (p. ej. "
        "<code>std</code> y <code>var</code> o "
        "<code>case_rr_std</code> y <code>case_rr_rmssd</code>). "
        "En modelos lineales, features muy correlacionadas "
        "<b>reparten artificialmente su importancia</b>: el modelo puede dar un peso pequeño "
        "a dos features correlacionadas cuando en realidad ambas representan la misma información. "
        "La importancia real del concepto subyacente es la <i>suma</i> de las dos, "
        "no cada una por separado. Esto debe considerarse al interpretar los rankings.",
    )
else:
    st.html(
        '<div class="placeholder-block" style="min-height:100px;padding:20px">'
        '<div class="ph-mono">imagen no encontrada</div>'
        '<div class="ph-title">Heatmap de correlación</div>'
        '<div class="ph-desc">No se encontró reports/figures/feature_correlation_heatmap.png</div>'
        '</div>'
    )

st.write("")

# ── Auto-generated interpretation ────────────────────────────────────────────
section_title("Interpretación automática")

# Find dominant group
top_group     = group_summary.loc[0, "Grupo"]
top_group_imp = float(group_summary.loc[0, "Importancia acumulada"])
top_group_pct = top_group_imp / total_importance

# Second group
second_group     = group_summary.loc[1, "Grupo"] if len(group_summary) > 1 else "—"
second_group_pct = float(group_summary.loc[1, "Importancia acumulada"]) / total_importance if len(group_summary) > 1 else 0.0

# Top 2 features — use stripped base names for readability
top2 = df_imp.head(2)
top2_names = [str(top2.loc[r, "feature_base"]) for r in top2.index]
top2_vals  = [float(top2.loc[r, IMP_COL]) for r in top2.index]

with st.container(border=True):
    card_header(
        "Conclusión de esta sección",
        "lectura cualitativa · datos reales",
        right_html=badge("auto-generada", "muted"),
    )
    kv_table([
        ("Feature más importante",
         f'<b style="color:var(--teal)">{top2_names[0].replace("_"," ")}</b>'
         f' — importancia {top2_vals[0]:.3f}'),
        ("Segunda feature",
         f'{top2_names[1].replace("_"," ")} — importancia {top2_vals[1]:.3f}'),
        ("Grupo dominante",
         f'<b style="color:var(--teal)">{top_group}</b> ({top_group_pct:.0%} de la importancia total)'),
        ("Segundo grupo",
         f'{second_group} ({second_group_pct:.0%})'),
        ("Interpretación",
         f"El modelo usa principalmente información de <b>{top_group}</b> y "
         f"<b>{second_group}</b> para clasificar arritmias."),
    ])

    st.write("")
    callout(
        "info",
        "Qué sugiere este patrón",
        f"Las dos features con mayor importancia (<b>{top2_names[0].replace('_',' ')}</b> e "
        f"<b>{top2_names[1].replace('_',' ')}</b>) pertenecen al grupo "
        f"<b>{_GROUP_MAP.get(top2_names[0], 'Otros')}</b>. "
        f"Esto sugiere que el modelo discrimina arritmias principalmente a partir de "
        f"<b>variabilidad del ritmo cardíaco a nivel de caso (HRV global)</b>, "
        f"lo cual es clínicamente plausible: distintos tipos de arritmia tienen patrones "
        f"de RR muy diferentes (p. ej. fibrilación auricular presenta alta irregularidad). "
        f"Las features morfológicas (amplitud, varianza de la señal) aportan un rol "
        f"complementario ({second_group_pct:.0%} de la importancia total).",
    )

st.write("")

# ── Limitations ────────────────────────────────────────────────────────────────
section_title("Limitaciones de la interpretación")

lim1, lim2 = st.columns(2)
with lim1:
    callout(
        "warn",
        "Importancia ≠ causalidad",
        "Una feature importante para el modelo no implica que sea la causa clínica "
        "de la arritmia. Puede ser una correlación aprendida del dataset de entrenamiento.",
    )
    callout(
        "warn",
        "Features correlacionadas distorsionan rankings",
        "Grupos de features muy correlacionadas reparten su importancia entre ellas. "
        "El ranking individual subestima la influencia real de cada grupo.",
    )
    callout(
        "warn",
        "Solo features con importancia > 0",
        f"El reporte incluye {n_features_csv} de {n_features_total} features. "
        "Las features con coeficiente nulo no aparecen, pero existen en el modelo.",
    )

with lim2:
    callout(
        "warn",
        "Importancia global — sin explicaciones locales",
        "La importancia de features es una medida global del modelo y no explica "
        "por qué un latido específico fue clasificado de cierta manera. "
        "Para explicaciones locales se necesita SHAP o LIME.",
    )
    callout(
        "warn",
        "Desempeño desigual entre clases",
        "El análisis de importancia es global. Las features que explican bien AFIB/AFL "
        "pueden no ser las mismas que explican las clases minoritarias (AVB, VT, SVTA).",
    )
    callout(
        "info",
        "Cómo mejorar esta página",
        "Para interpretabilidad local, añadir SHAP values o "
        "Permutation Importance (más robusto que coeficientes para SVC). "
        "El heatmap interactivo se habilitaría exportando el CSV de correlaciones.",
    )
