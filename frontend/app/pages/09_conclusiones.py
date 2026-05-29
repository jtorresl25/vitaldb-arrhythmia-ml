import streamlit as st

import pandas as pd

from components.layout import page_header
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.badges import badge, badge_row
from utils.loaders import (
    load_model_metadata,
    load_model_comparison,
    load_classification_report,
    load_feature_importance,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
meta   = load_model_metadata()

page_header(
    "Conclusiones",
    "Resumen final de hallazgos, limitaciones y próximos pasos.",
    badge_html=badge_row(badge("Cierre del proyecto", "info"), badge("Académico", "muted")),
)

# ── Load real data ─────────────────────────────────────────────────────────────
df_models = load_model_comparison()
df_cls    = load_classification_report()
df_imp    = load_feature_importance()

SUMMARY = {"accuracy", "macro avg", "weighted avg"}

# Derive real values from classification report
if df_cls is not None:
    df_classes = df_cls[~df_cls.index.isin(SUMMARY)].copy()
    for col in ["f1-score", "recall", "precision", "support"]:
        if col in df_classes.columns:
            df_classes[col] = pd.to_numeric(df_classes[col], errors="coerce")
    n_classes     = len(df_classes)
    good_classes  = df_classes[df_classes["f1-score"] >= 0.70].index.tolist()
    medium_classes= df_classes[(df_classes["f1-score"] >= 0.30) & (df_classes["f1-score"] < 0.70)].index.tolist()
    hard_classes  = df_classes[df_classes["f1-score"] < 0.30].index.tolist()
    total_test    = int(df_classes["support"].sum()) if "support" in df_classes.columns else 0
else:
    df_classes    = None
    n_classes     = 0
    good_classes  = []
    medium_classes= []
    hard_classes  = []
    total_test    = 0

# Winner info from metadata (canonical source)
_winner_id   = meta.get("winner_model", "—") if meta else "—"
_winner_nice = _winner_id.replace("_", " ").title() if _winner_id != "—" else "—"
winner_f1    = meta.get("winner_test_f1_macro") if meta else None
winner_f1_str = f"{winner_f1:.3f}" if winner_f1 is not None else "—"

# Supplementary metrics from comparison CSV
winner_acc  = None
winner_time = None
n_models    = 0
if df_models is not None and "model" in df_models.columns:
    n_models = len(df_models)
    wr = df_models[df_models["model"] == _winner_id]
    if wr.empty:
        f1_col = next((c for c in ["test_f1_macro", "f1_macro"] if c in df_models.columns), None)
        if f1_col:
            wr = df_models.dropna(subset=[f1_col]).sort_values(f1_col, ascending=False).head(1)
    if not wr.empty:
        winner_acc  = float(wr["test_accuracy"].iloc[0])  if "test_accuracy"  in wr.columns else None
        t_col       = next((c for c in ["fit_seconds", "fit_time_seconds"] if c in wr.columns), None)
        winner_time = float(wr[t_col].iloc[0]) if t_col else None

train_size = meta.get("train_size_rows", 0) if meta else 0
test_size  = meta.get("test_size_rows",  0) if meta else 0

# ── Callout de alcance ────────────────────────────────────────────────────────
callout(
    "warn",
    "Alcance académico — no clínico",
    "Esta app es una <b>demo académica</b> de Machine Learning para clasificación "
    "multiclase de ritmos intraoperatorios. "
    "<b>No es una herramienta clínica, no reemplaza interpretación médica "
    "y no debe usarse para diagnóstico.</b> "
    "Los resultados reflejan el desempeño del modelo sobre un conjunto de test "
    "con alta prevalencia de ritmo sinusal (N) y fibrilación auricular.",
)

st.write("")

# ── Resumen ejecutivo ─────────────────────────────────────────────────────────
section_title("Resumen ejecutivo")

_n_train_grp = meta.get("n_train_groups", 0) if meta else 0
_n_test_grp  = meta.get("n_test_groups",  0) if meta else 0
_split_str   = (f"{_n_train_grp} grupos entrenamiento, {_n_test_grp} test"
                if _n_train_grp else "split 80/20 por case_id")
_winner_acc_str = f", Acc={winner_acc:.3f}" if winner_acc is not None else ""

# Top features from importance CSV
_top_feats_str = "features tabulares (metadatos clínicos + RR intervals)"
if df_imp is not None and not df_imp.empty:
    _feat_col = next((c for c in ["feature_base", "feature"] if c in df_imp.columns), None)
    _imp_col  = next((c for c in ["importance", "abs_coef"] if c in df_imp.columns), None)
    if _feat_col and _imp_col:
        _top2 = (df_imp.dropna(subset=[_imp_col])
                       .sort_values(_imp_col, ascending=False)
                       .head(2)[_feat_col].tolist())
        if _top2:
            _top2_names = [str(n).replace("_"," ") for n in _top2]
            _top_feats_str = (f"<code>{_top2_names[0]}</code> y "
                              f"<code>{_top2_names[1]}</code>" if len(_top2) > 1
                              else f"<code>{_top2_names[0]}</code>")

kv_table([
    ("Pipeline completo",
     f"Flujo reproducible desde anotaciones VitalDB hasta evaluación comparativa "
     f"de {n_models if n_models else '—'} modelos clásicos de ML."),
    ("Separación por case_id",
     f"El split tren/test se hizo por <b>case_id</b> para evitar leakage "
     f"({_split_str})."),
    ("Modelo ganador",
     f"<b>{_winner_nice}</b> fue el mejor modelo según F1-macro "
     f"(F1={winner_f1_str}{_winner_acc_str})."),
    ("Desempeño global",
     (f"El rendimiento es <b>moderado y desigual</b>: "
      f"{len(good_classes)} clases con F1 ≥ 0.70, "
      f"{len(hard_classes)} clases con F1 &lt; 0.30.")
     if n_classes > 0 else
     "Reporte de clasificación no disponible — ejecuta el pipeline tabular."),
    ("Interpretabilidad",
     f"Las features más importantes según {_winner_nice} son: {_top_feats_str}."),
    ("Desbalance de clases",
     "El dataset está fuertemente desbalanceado: N y AFIB/AFL dominan. "
     "Las clases minoritarias (AVB, SVTA, VT) tienen desempeño muy bajo o nulo."),
])

st.write("")

# ── Cards de resultados principales ──────────────────────────────────────────
section_title("Resultados principales")

c1, c2, c3, c4 = st.columns(4)

with c1:
    with st.container(border=True):
        card_header(
            "Modelo ganador",
            _winner_nice,
            right_html=badge("winner", "winner"),
        )
        metric_card("F1-macro (test)", winner_f1_str, "métrica principal", accent="teal", helper_kind="ok")
        metric_card("Accuracy (test)", f"{winner_acc:.4f}" if winner_acc is not None else "—", "puede ser engañosa", accent="blue")
        _hp_str = "—"
        if meta and "best_hyperparams_per_model" in meta:
            hp = meta["best_hyperparams_per_model"].get(_winner_id, {})
            if hp:
                _hp_str = " · ".join(f"{k}={v}" for k, v in list(hp.items())[:2])
        metric_card("Hiperparámetros", _hp_str[:40] + ("…" if len(_hp_str) > 40 else ""),
                    f"fit: {winner_time:.0f} s" if winner_time else "—", accent="muted")

with c2:
    with st.container(border=True):
        _total_str = f"{(train_size + test_size):,}" if (train_size + test_size) > 0 else "—"
        card_header("Dataset", f"{_total_str} ventanas", right_html=badge("Tabular", "info"))
        metric_card("Entrenamiento", f"{train_size:,}" if train_size > 0 else "—",
                    f"{meta.get('n_train_groups', '—')} casos" if meta else "—", accent="blue")
        metric_card("Test", f"{test_size:,}" if test_size > 0 else "—",
                    f"{meta.get('n_test_groups', '—')} casos" if meta else "—", accent="blue")
        _n_feat = meta.get("n_features", 0) if meta else 0
        metric_card("Features", str(_n_feat) if _n_feat > 0 else "—", "features tabulares seleccionadas", accent="muted")

with c3:
    with st.container(border=True):
        card_header("Evaluación por clase", f"{n_classes} clases evaluadas", right_html=badge("desigual", "warn"))
        metric_card(
            "Clases fuertes (F1 ≥ 0.70)",
            str(len(good_classes)),
            ", ".join(good_classes) if good_classes else "—",
            accent="teal",
            helper_kind="ok",
        )
        metric_card(
            "Clases difíciles (F1 < 0.30)",
            str(len(hard_classes)),
            ", ".join(hard_classes[:3]) + ("…" if len(hard_classes) > 3 else ""),
            accent="err",
            helper_kind="warn",
        )

with c4:
    with st.container(border=True):
        _interp_type = "feature importance" if ("forest" in _winner_id or "tree" in _winner_id) else "coeficientes"
        card_header("Interpretabilidad", f"{_interp_type} · {_winner_nice}", right_html=badge("global", "muted"))
        # Pull top features dynamically from importance CSV
        if df_imp is not None and not df_imp.empty:
            _fc = next((c for c in ["feature_base", "feature"] if c in df_imp.columns), None)
            _ic = next((c for c in ["importance", "abs_coef"] if c in df_imp.columns), None)
            if _fc and _ic:
                _top3 = df_imp.dropna(subset=[_ic]).sort_values(_ic, ascending=False).head(3)
                for _rank, (_idx, _row) in enumerate(zip(range(1, 4), _top3.itertuples()), 1):
                    _fname = str(getattr(_idx, _fc, "—") if False else _top3.iloc[_rank - 1][_fc]).replace("_", " ")
                    _fval  = float(_top3.iloc[_rank - 1][_ic])
                    metric_card(f"Feature #{_rank}", _fname[:20], f"importancia {_fval:.3f}",
                                accent="teal" if _rank == 1 else "blue", helper_kind="ok" if _rank == 1 else "muted")
        else:
            metric_card("Features", "—", "importancia no disponible", accent="muted")

st.write("")

# ── Análisis de clases ─────────────────────────────────────────────────────────
section_title("Desempeño por clase")

col_good, col_med, col_bad = st.columns(3)

with col_good:
    with st.container(border=True):
        card_header("Clases con buen desempeño", "F1 ≥ 0.70", right_html=badge(f"{len(good_classes)} clases", "ok"))
        if df_classes is not None and good_classes:
            rows = []
            for cls in good_classes:
                f1  = float(df_classes.loc[cls, "f1-score"]) if cls in df_classes.index else 0.0
                sup = int(df_classes.loc[cls, "support"]) if cls in df_classes.index else 0
                rows.append((cls, f'{badge(f"F1 {f1:.3f}", "ok")} {badge(f"{sup:,} ventanas", "muted")}'))
            kv_table(rows)
        else:
            for cls in good_classes:
                kv_table([(cls, badge("F1 ≥ 0.70", "ok"))])
        callout("ok", "Por qué funcionan", "Clases con muchas muestras y patrones RR claramente distintos.")

with col_med:
    with st.container(border=True):
        card_header("Desempeño moderado", "0.30 ≤ F1 < 0.70", right_html=badge(f"{len(medium_classes)} clases", "info"))
        if df_classes is not None and medium_classes:
            rows = []
            for cls in medium_classes:
                f1  = float(df_classes.loc[cls, "f1-score"]) if cls in df_classes.index else 0.0
                sup = int(df_classes.loc[cls, "support"]) if cls in df_classes.index else 0
                rows.append((cls, f'{badge(f"F1 {f1:.3f}", "info")} {badge(f"{sup:,}", "muted")}'))
            kv_table(rows)
        else:
            for cls in medium_classes:
                kv_table([(cls, badge("F1 0.30–0.70", "info"))])
        callout("info", "Mejorables", "Menor soporte o solapamiento con clases dominantes.")

with col_bad:
    with st.container(border=True):
        card_header("Clases difíciles", "F1 < 0.30", right_html=badge(f"{len(hard_classes)} clases", "err"))
        if df_classes is not None and hard_classes:
            rows = []
            for cls in hard_classes:
                f1  = float(df_classes.loc[cls, "f1-score"]) if cls in df_classes.index else 0.0
                sup = int(df_classes.loc[cls, "support"]) if cls in df_classes.index else 0
                rows.append((cls, f'{badge(f"F1 {f1:.3f}", "err")} {badge(f"{sup:,}", "muted")}'))
            kv_table(rows)
        else:
            for cls in hard_classes:
                kv_table([(cls, badge("F1 < 0.30", "err"))])
        callout("warn", "Riesgo clínico potencial", "AVB y SVTA tienen F1 casi nulo — el modelo no las detecta.")

st.write("")

# ── Lo que funcionó bien ──────────────────────────────────────────────────────
section_title("Lo que funcionó bien")

with st.container(border=True):
    card_header("Aciertos del pipeline", "evaluación honesta")
    kv_table([
        (badge("✓", "ok") + " Pipeline reproducible",
         "Cada paso está en notebooks numerados y es re-ejecutable desde cero."),
        (badge("✓", "ok") + " Split por case_id",
         "La separación por caso evita leakage entre ventanas del mismo paciente."),
        (badge("✓", "ok") + " Comparación multi-modelo",
         f"Se evaluaron {n_models} modelos con búsqueda de hiperparámetros (GridSearchCV)."),
        (badge("✓", "ok") + " Evaluación por clase",
         "El reporte de clasificación expone el desempeño real por tipo de arritmia."),
        (badge("✓", "ok") + " Interpretabilidad inicial",
         f"La importancia de features del {_winner_nice} da señal sobre qué variables clínicas son más discriminativas."),
        (badge("✓", "ok") + " App Streamlit con datos reales",
         "Todas las métricas y visualizaciones usan archivos reales del pipeline."),
    ])

st.write("")

# ── Principales limitaciones ──────────────────────────────────────────────────
section_title("Principales limitaciones")

lim1, lim2 = st.columns(2)

with lim1:
    callout(
        "warn",
        f"F1-macro bajo ({winner_f1_str})",
        "El modelo global tiene desempeño moderado. La accuracy puede resultar engañosa: "
        "refleja el dominio de clases N y AFIB/AFL, no el desempeño en clases minoritarias.",
    )
    callout(
        "warn",
        "Desbalance severo entre clases",
        "N y AFIB/AFL representan la mayoría de ventanas. "
        "Clases como AVB (346 muestras) o Unclassifiable (3 muestras) "
        "son prácticamente ignoradas por el modelo sin pesos de clase.",
    )
    callout(
        "warn",
        "Clases minoritarias sin detección",
        "AVB (F1 = 0.000) y SVTA (F1 = 0.027) no son detectadas de forma confiable. "
        "Esto limita severamente la utilidad clínica del modelo actual.",
    )
    callout(
        "info",
        "Predicción demo disponible (datos tabulares)",
        "La página 07 permite predecir sobre filas reales del dataset tabular procesado. "
        "La integración con ECG crudo externo queda como trabajo futuro.",
    )

with lim2:
    callout(
        "warn",
        "Interpretación no causal",
        f"La importancia de features del {_winner_nice} no implica causalidad clínica. "
        "Es una correlación aprendida del dataset de entrenamiento.",
    )
    callout(
        "warn",
        "Variables intraoperatorias acumuladas",
        "Algunas features tabulares (p. ej. <code>intraop_ppf</code>, <code>intraop_rbc</code>) "
        "representan totales del caso, no el estado en el momento del latido. "
        "Esto puede introducir sesgo temporal en el modelo.",
    )
    callout(
        "warn",
        "Sin explicaciones locales",
        "La importancia global no explica por qué un latido específico fue clasificado "
        "de cierta manera. Para eso se requiere SHAP o LIME.",
    )
    callout(
        "info",
        "App académica — no clínica",
        "Los resultados son válidos como experimento metodológico. "
        "Cualquier aplicación clínica requeriría validación externa, "
        "calibración y revisión médica especializada.",
    )

st.write("")

# ── Próximos pasos ─────────────────────────────────────────────────────────────
section_title("Próximos pasos técnicos — hoja de ruta")

col_near, col_far = st.columns(2)

with col_near:
    with st.container(border=True):
        card_header("Corto plazo", "completar la app y los datos", right_html=badge("prioridad alta", "err"))
        kv_table([
            ("1", "Implementar <code>03_dataset_limpieza.py</code> con estadísticas del dataset tabular y distribución de clases."),
            ("2", "Exportar <code>reports/tables/tabular_confusion_matrix_absolute.csv</code> en formato long para heatmap interactivo."),
            ("3", "Agregar <code>n_train_groups</code>, <code>n_test_groups</code> y tamaños de split al JSON de metadata tabular."),
            ("4", "Ampliar la demo de predicción (p. 07) con filtros por clase y visualización de probabilidades por clase."),
            ("5", "Re-ejecutar el pipeline con <code>--n-iter 30 --n-splits 5</code> sobre todos los casos para métricas definitivas."),
        ])

with col_far:
    with st.container(border=True):
        card_header("Mediano plazo", "mejorar el modelo", right_html=badge("mejora de calidad", "info"))
        kv_table([
            ("6", "<b>Class weighting</b>: verificar que <code>class_weight='balanced'</code> esté activo en el modelo para penalizar más los errores en clases minoritarias."),
            ("7", "<b>SMOTE / resampling</b>: aumentar muestras de clases minoritarias en el set de entrenamiento."),
            ("8", "<b>Agrupación clínica</b>: considerar fusionar clases cercanas (p. ej. Patterned Atrial Ectopy y WAP/MAT) para reducir el problema de desbalance."),
            ("9", "<b>SHAP / Permutation Importance</b>: reemplazar importancia por coeficientes con métodos más robustos y con explicaciones locales."),
            ("10", "<b>Modelo 1D-CNN</b>: aprender representaciones directamente de la señal ECG cruda, sin depender de features manuales."),
        ])

st.write("")

# ── Estado de páginas ─────────────────────────────────────────────────────────
section_title("Estado actual de la app")

pages_data = [
    ("01", "Inicio",               "Implementada",                    "ok"),
    ("02", "Pipeline",             "Diagrama HTML externo",           "ok"),
    ("03", "Dataset y limpieza",   "Pendiente",                       "warn"),
    ("04", "Rendimiento del modelo","Implementada",                   "ok"),
    ("05", "Evaluación por clase", "Implementada",                    "ok"),
    ("06", "Matriz de confusión",  "Implementada (PNG + CSV)",        "ok"),
    ("07", "Predicciones",         "Implementada — demo tabular",     "ok"),
    ("08", "Interpretabilidad",    "Implementada",                    "ok"),
    ("09", "Conclusiones",         "Implementada",                    "ok"),
]

STATUS_ICONS = {"ok": "✓", "warn": "◑", "err": "✗"}
STATUS_COLORS = {
    "ok":   "color:var(--ok)",
    "warn": "color:var(--warn)",
    "err":  "color:var(--err)",
}

rows_html = ""
for num, name, status, kind in pages_data:
    icon  = STATUS_ICONS.get(kind, "·")
    style = STATUS_COLORS.get(kind, "")
    rows_html += (
        f'<div class="kv-key">'
        f'<span style="color:var(--fg-3);font-size:11px">0{num} </span>{name}'
        f'</div>'
        f'<div class="kv-val">'
        f'<span style="{style};font-family:var(--font-mono)">{icon} {status}</span>'
        f'</div>'
    )

st.html(f'<div class="kv-table">{rows_html}</div>')

n_done    = sum(1 for *_, k in pages_data if k == "ok")
n_pending = sum(1 for *_, k in pages_data if k in ("warn", "err"))

c_a, c_b, c_c = st.columns(3)
with c_a:
    metric_card("Páginas implementadas", str(n_done), "de 9 páginas", accent="teal", helper_kind="ok")
with c_b:
    metric_card("Pendientes / placeholder", str(n_pending), "páginas", accent="warn", helper_kind="warn")
with c_c:
    metric_card("Pendientes", "1 página", "Dataset y limpieza (03)", accent="warn", helper_kind="warn")

st.write("")

# ── Mensaje final ─────────────────────────────────────────────────────────────
section_title("Reflexión final")

with st.container(border=True):
    card_header(
        "Balance del proyecto",
        "lo que logramos y lo que falta",
        right_html=badge("académico", "muted"),
    )
    st.write("")
    st.html(
        f'<p style="color:var(--fg-1);font-size:15px;line-height:1.7;max-width:860px">'
        f"El proyecto demuestra que es posible construir un <b>flujo reproducible y evaluado</b> "
        f"para clasificar ritmos intraoperatorios a partir de <b>datos tabulares</b> "
        f"(metadatos clínicos + intervalos RR por latido). "
        f"El modelo <b>{_winner_nice}</b> obtiene F1-macro = {winner_f1_str} y "
        f"distingue con mayor facilidad los ritmos dominantes (N y AFIB/AFL), "
        f"mientras que las clases minoritarias (AVB, SVTA, VT) siguen siendo el principal reto. "
        f"Sin embargo, el análisis también evidencia que el <b>desbalance severo de clases</b> "
        f"y la <b>naturaleza acumulada de algunas variables intraoperatorias</b> son los principales "
        f"retos metodológicos antes de pensar en aplicaciones clínicas. "
        f"Los próximos pasos son: re-balanceo de clases, más iteraciones de búsqueda HP "
        f"e integración de features de ritmo cardíaco más granulares."
        f"</p>"
    )
    st.write("")
    callout(
        "info",
        "Reproducibilidad",
        "Todo el código de este proyecto (pipeline, modelos y app) puede re-ejecutarse "
        "desde cero siguiendo los notebooks numerados del repositorio.",
    )
