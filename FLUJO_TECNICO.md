# Resumen final del flujo de procesamiento y entrenamiento
> Guía operativa para construir el notebook explicativo final.
> Basada exclusivamente en los archivos del repositorio verificados al 2026-05-30.

---

## 1. Objetivo final del proyecto

El proyecto clasifica latidos cardíacos intraoperatorios como **Normal** o **Anormal** usando datos de la base de datos VitalDB Arrhythmia (PhysioNet). Cada latido tiene una etiqueta original multiclase (`rhythm_label`), pero el flujo final colapsa todas las clases en dos grupos:

- **Normal** → `rhythm_label == "N"`
- **Anormal** → cualquier otro valor de `rhythm_label`

El modelo **no usa señal ECG cruda**. Usa exclusivamente features tabulares: intervalos RR derivados de las anotaciones y metadatos clínicos del paciente/cirugía presentes en `metadata.csv`.

El proyecto tuvo una fase multiclase inicial (flujo legacy en `src/search.py`, `src/windowing.py` y notebooks `03`–`06_full_modeling_hyperparameter_search.ipynb`), pero ese flujo fue descartado y reemplazado por el flujo tabular activo. El notebook final debe centrar la explicación en el modelo binario.

---

## 2. Archivos principales del repositorio

| Archivo / carpeta | Rol en el proyecto | Entrada | Salida | Usado en script | Usado en app |
|---|---|---|---|---|---|
| `data/raw/physionet_annotations/metadata.csv` | Metadatos del paciente y cirugía | — | — | scripts 01–03 | No directo |
| `data/raw/physionet_annotations/Annotation_Files/Annotation_file_*.csv` | Anotaciones latido a latido (494 archivos) | — | — | scripts 01–02 | No directo |
| `data/processed/filtered_tabular_modeling_dataset.parquet` | Dataset tabular filtrado y enriquecido (~640 k filas) | Anotaciones + metadata | — | script 03 | Sí (predicción local) |
| `scripts/01_audit_filtered_tabular_dataset.py` | Auditoría de calidad del dataset | Anotaciones + metadata | CSVs de auditoría | — | No |
| `scripts/02_build_filtered_tabular_modeling_dataset.py` | Construye el parquet principal | Anotaciones + metadata | `filtered_tabular_modeling_dataset.parquet` | — | No |
| `scripts/03_run_tabular_binary_hyperparameter_search_FIXED.py` | Entrenamiento binario + búsqueda de hiperparámetros | parquet | modelo .joblib + reportes | — | No |
| `scripts/04_prepare_streamlit_artifacts.py` | Copia artefactos a `frontend/app/app_artifacts/` | modelos + reportes | app_artifacts/ | — | No |
| `scripts/05_select_binary_demo_cases.py` | Selecciona casos demo para la app | parquet + modelo | `binary_demo_case_candidates.csv` | — | No |
| `scripts/06_prepare_selected_demo_npy.py` | Extrae fragmentos .npy para la demo | waveforms + demo cases | fragmentos .npy | — | No |
| `src/config.py` | Rutas, nombres de columnas, parámetros globales | — | — | todos | Sí |
| `src/data_loading.py` | Carga metadata + anotaciones, merge | — | — | script 02 | No |
| `src/preprocessing.py` | Filtros básicos + preprocesador ColumnTransformer | — | — | scripts 02–03 | No |
| `src/modeling.py` | Split por grupo + validaciones | — | — | script 03 | No |
| `src/tabular_search.py` | Pipeline de búsqueda + clasificadores | — | — | script 03 | No |
| `src/evaluation.py` | Métricas, matrices de confusión, reportes | — | — | script 03 | No |
| `src/features.py` | Features de RR (funciones auxiliares) | — | — | script 02 | No |
| `models/tabular_best_model_pipeline.joblib` | Modelo entrenado (LinearSVC binario) | — | — | script 04 | Sí |
| `models/tabular_best_model_metadata.json` | Metadatos del modelo (features, hiperparámetros, métricas) | — | — | script 04 | Sí |
| `reports/tables/tabular_binary_metrics.csv` | Métricas binarias finales del test | — | — | — | Sí |
| `reports/tables/tabular_best_model_classification_report.csv` | Reporte por clase (precision/recall/F1) | — | — | — | Sí |
| `reports/tables/tabular_confusion_matrix_absolute.csv` | Matriz de confusión absoluta | — | — | — | Sí |
| `reports/tables/tabular_feature_importance_best_model.csv` | Importancia de features (coeficientes LinearSVC) | — | — | — | Sí |
| `reports/tables/tabular_feature_list_used.csv` | Lista exacta de features usadas | — | — | — | Sí |
| `reports/tables/tabular_train_test_split_summary.csv` | Resumen del split train/test | — | — | — | Sí |
| `reports/tables/tabular_model_comparison_test.csv` | Comparativa de modelos en test | — | — | — | Sí |
| `reports/tables/tabular_model_comparison_history.csv` | Historial de corridas (exploratorias + final) | — | — | — | Sí |
| `reports/tables/binary_case_level_metrics.csv` | Métricas por caso demo | — | — | script 05 | Sí |
| `reports/tables/binary_demo_case_candidates.csv` | Casos demo seleccionados | — | — | script 05 | Sí |
| `reports/figures/tabular_best_model_confusion_matrix_absolute.png` | Imagen de la matriz de confusión | — | — | — | Sí |
| `reports/figures/tabular_best_model_confusion_matrix_normalized.png` | Imagen de la matriz normalizada | — | — | — | Sí |
| `frontend/app/app_artifacts/models/` | Copia del modelo + metadata para Streamlit Cloud | — | — | — | Sí |
| `frontend/app/app_artifacts/reports/tables/` | Copia de CSVs de reporte para Streamlit Cloud | — | — | — | Sí |
| `frontend/app/app_artifacts/demo/npy_cases/case_*.npy` | Fragmentos ECG demo (casos 337, 2040, 5377, 3098) | — | — | — | Sí |
| `frontend/app/app_artifacts/demo/case_features/case_*.parquet` | Features pre-procesadas por caso demo | — | — | — | Sí |
| `frontend/app/app_artifacts/demo/demo_cases_binary.csv` | Catálogo de casos demo con descripción | — | — | — | Sí |
| `frontend/app/app.py` | Punto de entrada Streamlit | — | — | — | Sí |
| `frontend/app/pages/07_predicciones.py` | Página de predicción/demo | — | — | — | Sí |
| `frontend/app/utils/loaders.py` | Carga modelo, metadata y CSVs en la app | — | — | — | Sí |
| `requirements.txt` / `environment.yml` | Dependencias del proyecto | — | — | — | Sí |
| `.gitignore` | Define qué NO se sube (data/, models/, parquets, joblibs, etc.) | — | — | — | No |

---

## 3. Flujo paso a paso para construir el notebook

### Paso 1. Cargar datos originales

**Archivo de entrada principal:** `data/raw/physionet_annotations/metadata.csv`
**Anotaciones:** 494 archivos `data/raw/physionet_annotations/Annotation_Files/Annotation_file_<case_id>.csv`

El flujo parte desde:
1. `metadata.csv` — metadatos del paciente y cirugía (age, sex, height, weight, bmi, asa, department, optype, approach, position, ane_type, preop_labs, intraop_drugs, etc.)
2. Archivos de anotación por caso — una fila por latido con columnas: `case_id`, `time_second`, `rhythm_label`, `beat_type`, `bad_signal_quality`.

El merge se hace por `case_id` (inner join). Función en `src/data_loading.py`:
```python
from src.data_loading import load_all_annotations, load_metadata, merge_metadata_and_annotations
annotations = load_all_annotations()
metadata    = load_metadata()
merged      = merge_metadata_and_annotations(metadata, annotations, on="case_id", how="inner")
```

**Columnas clave obligatorias después del merge:**
`case_id`, `time_second`, `rhythm_label`, `beat_type`, `bad_signal_quality`  
+ todas las columnas de `metadata.csv` (age, sex, height, weight, optype, etc.)

**Formato de patrón de archivos de anotación** (de `src/config.py`):
```
ANNOTATION_FILENAME_REGEX = r"^Annotations?_file_(\d+)\.csv$"
```

---

### Paso 2. Limpieza y validación

Función en `src/preprocessing.py`: `apply_basic_filters(df)`

Filtros que se aplican (en orden):
1. **Excluir clase "Noise"**: `rhythm_label != "Noise"` (ver `config.EXCLUDED_RHYTHM_LABELS`)
2. **Excluir mala calidad**: `bad_signal_quality == False` (o 0)
3. **Eliminar `rhythm_label` nulos** o strings vacíos / "nan" / "none": función `_drop_label_nans` en `scripts/02_build_filtered_tabular_modeling_dataset.py`
4. **Coerción de `age`**: valores `">89"` → `89.0` (hay pacientes anonimizados en PhysioNet)
5. **Eliminar columnas constantes** (todo NaN o un solo valor): se descartan del parquet para reducir tamaño

No hay una función explícita de deduplicación, pero el merge inner garantiza que solo entran registros con case_id en ambas fuentes.

---

### Paso 3. Construcción de features

Las features se construyen en `scripts/02_build_filtered_tabular_modeling_dataset.py`, función `_add_within_case_time_features(df)`.

**Features derivadas por latido dentro de cada caso** (calculadas localmente por `case_id`):

| Feature | Descripción | Tipo |
|---|---|---|
| `rr_prev` | Diferencia de tiempo (segundos) al latido anterior | Numérica |
| `rr_next` | Diferencia de tiempo (segundos) al latido siguiente | Numérica |
| `hr_inst_from_rr_prev` | Frecuencia cardíaca instantánea en bpm (60 / rr_prev) | Numérica |
| `position_in_case` | Posición relativa del latido en el caso (0.0 a 1.0) | Numérica |

**Importante:** estas features se calculan ordenando cada caso por `time_second` y usando `.diff()` por grupo para evitar cruzar latidos de un caso A con otro caso B.

**Features de metadatos** (vienen directamente de `metadata.csv`):

Numéricas (57 en total): `time_second`, `analysis_start_time_sec`, `analysis_end_time_sec`, `analyzed_duration_sec`, `total_beats`, `caseend`, `anestart`, `aneend`, `opstart`, `opend`, `age`, `height`, `weight`, `bmi`, `asa`, `emop`, `preop_htn`, `preop_dm`, `preop_hb`, `preop_plt`, `preop_pt`, `preop_aptt`, `preop_na`, `preop_k`, `preop_gluc`, `preop_alb`, `preop_ast`, `preop_alt`, `preop_bun`, `preop_cr`, `preop_ph`, `preop_hco3`, `preop_be`, `preop_pao2`, `preop_paco2`, `preop_sao2`, `tubesize`, `lmasize`, `intraop_ebl`, `intraop_uo`, `intraop_rbc`, `intraop_ffp`, `intraop_crystalloid`, `intraop_colloid`, `intraop_ppf`, `intraop_mdz`, `intraop_ftn`, `intraop_rocu`, `intraop_vecu`, `intraop_eph`, `intraop_phe`, `intraop_epi`, `intraop_ca`, + `rr_prev`, `rr_next`, `hr_inst_from_rr_prev`, `position_in_case`

Categóricas (16): `sex`, `department`, `optype`, `approach`, `position`, `ane_type`, `preop_ecg`, `preop_pft`, `cormack`, `dltubesize`, `iv1`, `iv2`, `aline1`, `aline2`, `cline1`, `cline2`

**Features más importantes** (top del modelo LinearSVC, según `tabular_feature_importance_best_model.csv`):

| Feature | Descripción |
|---|---|
| `preop_ecg_Premature atrial complexes` | ECG preoperatorio con complejos auriculares prematuros |
| `preop_ecg_Normal Sinus Rhythm` | ECG preoperatorio normal |
| `preop_pft_Severe restrictive` | Prueba función pulmonar severa restrictiva |
| `position_Reverse Trendelenburg` | Posición quirúrgica |
| `preop_ph` | pH preoperatorio arterial |
| `preop_paco2` | paCO2 preoperatorio |
| `iv2_Left hand` / `iv2_Right` | Acceso venoso |

**Nota metodológica crítica:** `beat_type` está presente en el parquet únicamente como referencia descriptiva. Está explícitamente prohibida como predictora (`config.TABULAR_LEAKAGE_COLUMNS`). Usarla como feature sería leakage directo porque determina el ritmo.

---

### Paso 4. Construcción del target binario

La transformación se implementa en `scripts/03_run_tabular_binary_hyperparameter_search_FIXED.py`, función `_make_binary_target` (líneas 134–142):

```python
def _make_binary_target(y: np.ndarray) -> np.ndarray:
    """Convierte rhythm_label multiclase a etiqueta binaria."""
    y_str = pd.Series(y).astype(str).str.strip()
    return np.where(y_str.eq("N"), "normal", "abnormal")
```

Regla exacta: `"N"` (cadena, sin espacios extra) → `"normal"`. Cualquier otra etiqueta → `"abnormal"`.

En la app Streamlit (`frontend/app/pages/07_predicciones.py`, líneas 55–59):
```python
def _to_binary(labels) -> np.ndarray:
    return np.array([
        "normal" if str(lbl).strip() == "N" else "abnormal"
        for lbl in np.asarray(labels).astype(str)
    ])
```

Clases originales que se agrupan como "anormal": todas las distintas de "N" (AFIB, AFL, VT, SVT, BB, etc., y cualquier etiqueta no-N del dataset).

---

### Paso 5. Separación train/test

Implementada en `src/modeling.py`, función `make_train_test_group_split_with_coverage`.

**Método:** `GroupShuffleSplit` (no `StratifiedGroupKFold`). El split se hace a nivel de `case_id` para garantizar que todos los latidos de un mismo paciente queden en un solo conjunto.

**Parámetros por defecto** (usados en la última corrida):
- `test_size = 0.20`
- `random_state = 42`
- `max_attempts = 200` (busca el seed que cubra ambas clases en test)

**Resultado confirmado** (de `tabular_train_test_split_summary.csv`):
- Train: 385 casos / 510,287 latidos
- Test: 97 casos / 129,173 latidos
- Fracción real de test: 20.2%

**Por qué no dividir aleatoriamente por fila:** un mismo caso puede tener latidos normales y anormales intercalados. Dividir por fila dejaría el modelo "ver" los latidos adyacentes del mismo paciente en entrenamiento, lo que inflaría métricas y no sería representativo del uso real (clasificar un paciente nuevo completo).

Verificación de no-leakage (en el script, línea 517):
```python
assert set(groups_train).isdisjoint(set(groups_test)), "Fuga de grupo."
```

---

### Paso 6. Preprocesamiento

Implementado en `src/preprocessing.py`, función `build_tabular_preprocessor`.

Pipeline: `ColumnTransformer` con dos ramas:
- **Numéricas:** `SimpleImputer(strategy="median")` → `StandardScaler()`
- **Categóricas:** `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)`

Parámetros clave (de `src/config.py`):
```python
TABULAR_MAX_CATEGORY_CARDINALITY = 30   # Categorías con >30 valores únicos se excluyen
TABULAR_OHE_MIN_FREQUENCY        = 50   # Categorías con <50 filas se agrupan en 'infrequent'
```

Columnas excluidas antes del preprocesador (de `config.TABULAR_LEAKAGE_COLUMNS`):
`rhythm_label`, `beat_type`, `rhythm_classes`, `bad_signal_quality`, `bad_signal_quality_label`, `case_id`, `caseid`, `subjectid`, `source_file`, `icu_days`, `death_inhosp`, `adm`, `dis`

La lista exacta de features usadas se guarda en `reports/tables/tabular_feature_list_used.csv`. La lista de columnas excluidas y su razón se guarda en `reports/tables/tabular_excluded_columns_leakage.csv`.

---

### Paso 7. Entrenamiento de modelos

#### 7a. Modelos multiclase iniciales (fase legacy — no es el resultado final)

Se entrenaron como multiclase (objetivo: `rhythm_label` sin colapsar). Los resultados son pobres por desbalance extremo de clases. Scripts legacy: `scripts/03_run_tabular_hyperparameter_search.py` y notebooks `05_baseline_modeling.ipynb`, `06_full_modeling_hyperparameter_search.ipynb`.

**Resultado multiclase confirmado en `models/README.md` y `reports/tables/model_comparison.csv`:**

| Modelo | test_f1_macro | test_accuracy | Tiempo |
|---|---|---|---|
| LinearSVC (ganador multiclase) | 0.3439 | 0.8061 | 185 s |
| Random Forest | 0.3225 | 0.7879 | 1467 s |
| MLP | 0.3065 | 0.8204 | 561 s |
| XGBoost | 0.2739 | 0.8203 | 640 s |

Estos números son bajos porque varias clases tienen muy pocas instancias (AFL, VT, SVT, BB son raras). El F1-macro penaliza fuertemente las clases sin soporte.

#### 7b. Benchmark exploratorio binario (150 casos, no resultado final)

Documentado en `reports/tables/tabular_model_comparison_history.csv`:

| Modelo | test_f1_macro | Tipo de corrida |
|---|---|---|
| MLP | 0.718 | exploratory_150_cases |
| LinearSVC | 0.641 | exploratory_150_cases |
| Random Forest | 0.624 | exploratory_150_cases |
| LogReg | 0.618 | exploratory_150_cases |
| Decision Tree | 0.556 | exploratory_150_cases |

**Nota:** MLP ganó el benchmark exploratorio pero fue descartado como modelo final por no ser interpretable y presentar advertencias de convergencia (según la nota en el CSV).

#### 7c. Modelo binario final (corrida completa, 482 casos)

Script: `scripts/03_run_tabular_binary_hyperparameter_search_FIXED.py`

**Ganador confirmado en `models/tabular_best_model_metadata.json`:**
- Modelo: `LinearSVC`
- Hiperparámetros: `C = 0.00195` (encontrado por `RandomizedSearchCV`, 30 iteraciones)
- Target: binario (`normal` vs `abnormal`)
- Entrenado el: 2026-05-30

**Importante: Los modelos de `logreg_balanced` con `C ≈ 0.0746` y las métricas de sensitivity=0.914, ROC-AUC=0.956 mencionadas en este documento no se encuentran en ningún archivo del repositorio.** El estado actual muestra LinearSVC como ganador con las métricas indicadas abajo. Es posible que esos resultados correspondan a una corrida anterior que no fue persistida o que el modelo entrenado con esos parámetros no fue guardado. Se recomienda re-ejecutar `scripts/03_run_tabular_binary_hyperparameter_search_FIXED.py` con `--models logreg` para reproducir y verificar.

El script usa `RandomizedSearchCV` con:
```python
n_iter = 30          # combinaciones por modelo
n_splits = 5         # folds de CV interna
cv = StratifiedGroupKFold (con fallback a GroupKFold)
scoring = "f1_macro"
```

---

### Paso 8. Evaluación

Función en `src/evaluation.py`. El test se evalúa **una sola vez** al final (el test está congelado durante la búsqueda).

**Métricas calculadas y guardadas** (para el modelo LinearSVC final):

Fuente: `reports/tables/tabular_binary_metrics.csv`

| Métrica | Valor (LinearSVC actual) |
|---|---|
| TN (Normal correcto) | 54,625 |
| FP (Normal predicho Anormal) | 24,286 |
| FN (Anormal predicho Normal) | 23,180 |
| TP (Anormal correcto) | 27,082 |
| Sensitivity / Recall Anormal | 0.539 |
| Specificity / Recall Normal | 0.692 |
| Precision Anormal | 0.527 |
| F1 Anormal | 0.533 |
| Accuracy | 0.633 |
| F1-macro (test) | 0.615 |

Fuente: `reports/tables/tabular_best_model_classification_report.csv`

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| abnormal | 0.527 | 0.539 | 0.533 | 50,262 |
| normal | 0.702 | 0.692 | 0.697 | 78,911 |
| macro avg | 0.615 | 0.616 | 0.615 | 129,173 |

**Métricas no calculadas actualmente (ausentes en los archivos):** ROC-AUC y Average Precision no están guardadas en ningún CSV. Solo están disponibles F1-macro, accuracy, precision, recall y balanced_accuracy. Si se quieren calcular, hay que modificar `_save_binary_extra_metrics` en el script para agregar `roc_auc_score` y `average_precision_score`.

**Discrepancia con los resultados reportados en la consigna de este documento:** Las métricas logreg_balanced (sensitivity=0.914, specificity=0.898, ROC-AUC=0.956, TN=8319, TP=12528) corresponden a un conjunto de test de solo 22,972 registros (vs 129,173 en la corrida actual). Esto indica que probablemente correspondían a una corrida con `--max-cases` o con un subconjunto diferente. Esos resultados **no están en ningún archivo del repositorio actual**.

---

### Paso 9. Archivos de salida generados

#### Por `scripts/02_build_filtered_tabular_modeling_dataset.py`
| Archivo | Contenido | Subir a GitHub |
|---|---|---|
| `data/processed/filtered_tabular_modeling_dataset.parquet` | Dataset tabular completo (~640 k filas) | **No** (en .gitignore, ~600 MB) |

#### Por `scripts/03_run_tabular_binary_hyperparameter_search_FIXED.py`
| Archivo | Contenido | Subir a GitHub |
|---|---|---|
| `models/tabular_best_model_pipeline.joblib` | Pipeline sklearn completo (preprocesador + LinearSVC) | **No** (en .gitignore) |
| `models/tabular_best_model_metadata.json` | Features, hiperparámetros, métricas, timestamp | **No** (en .gitignore) |
| `reports/tables/tabular_binary_metrics.csv` | TN, FP, FN, TP, sensitivity, specificity, precision, F1 | **No** (en .gitignore) |
| `reports/tables/tabular_best_model_classification_report.csv` | Precision/recall/F1 por clase | **No** |
| `reports/tables/tabular_confusion_matrix_absolute.csv` | Matriz de confusión en formato tabla | **No** |
| `reports/tables/tabular_feature_importance_best_model.csv` | Coeficientes del modelo por feature | **No** |
| `reports/tables/tabular_feature_list_used.csv` | Listado de features numéricas y categóricas usadas | **No** |
| `reports/tables/tabular_excluded_columns_leakage.csv` | Columnas excluidas y motivo | **No** |
| `reports/tables/tabular_train_test_split_summary.csv` | Resumen del split | **No** |
| `reports/tables/tabular_class_support_train_test.csv` | Soporte por clase en train y test | **No** |
| `reports/tables/tabular_model_comparison_cv.csv` | CV scores por modelo | **No** |
| `reports/tables/tabular_model_comparison_test.csv` | Test scores por modelo | **No** |
| `reports/tables/tabular_best_hyperparameters.csv` | Mejores hiperparámetros de cada modelo | **No** |
| `reports/tables/tabular_hyperparameter_search_meta.json` | Estado completo del run (JSON) | **No** |
| `reports/figures/tabular_best_model_confusion_matrix_absolute.png` | Imagen matriz absoluta | **No** |
| `reports/figures/tabular_best_model_confusion_matrix_normalized.png` | Imagen matriz normalizada | **No** |

#### Por `scripts/04_prepare_streamlit_artifacts.py`
| Archivo | Contenido | Subir a GitHub |
|---|---|---|
| `frontend/app/app_artifacts/models/tabular_best_model_pipeline.joblib` | Copia del modelo para la app | **Sí** (necesario para que la app funcione sin acceso a `models/`) |
| `frontend/app/app_artifacts/models/tabular_best_model_metadata.json` | Copia de metadata para la app | **Sí** |
| `frontend/app/app_artifacts/reports/tables/*.csv` | Copias de todos los CSVs de reporte | **Sí** |
| `frontend/app/app_artifacts/reports/figures/*.png` | Copias de matrices de confusión | **Sí** |
| `frontend/app/app_artifacts/demo/demo_cases_binary.csv` | Catálogo de casos demo | **Sí** |
| `frontend/app/app_artifacts/demo/case_features/case_*.parquet` | Features pre-computadas de los 4 casos demo | **Sí** |
| `frontend/app/app_artifacts/demo/npy_cases/case_*.npy` | Fragmentos ECG (4 casos: 337, 2040, 5377, 3098) | **Sí** (son fragmentos pequeños) |

**Importante:** `data/`, `models/` y `reports/` están en `.gitignore`. Para que la app funcione en Streamlit Cloud, **todos los artefactos necesarios deben copiarse a `frontend/app/app_artifacts/`** usando el script 04. Los archivos en `app_artifacts/` sí se suben a GitHub.

---

### Paso 10. Conexión con Streamlit

La app está en `frontend/app/app.py` y tiene 10 páginas en `frontend/app/pages/`.

**Archivos que necesita la app para funcionar correctamente:**

| Archivo | Ruta en app_artifacts | Cargado por |
|---|---|---|
| Modelo .joblib | `models/tabular_best_model_pipeline.joblib` | `utils/loaders.py:load_model()` |
| Metadata .json | `models/tabular_best_model_metadata.json` | `utils/loaders.py:load_model_metadata()` |
| Métricas binarias | `reports/tables/tabular_binary_metrics.csv` | `utils/loaders.py` |
| Reporte por clase | `reports/tables/tabular_best_model_classification_report.csv` | `pages/05_evaluacion_clase.py` |
| Comparativa modelos | `reports/tables/tabular_model_comparison_test.csv` | `pages/04_rendimiento_modelo.py` |
| Historial modelos | `reports/tables/tabular_model_comparison_history.csv` | `pages/04_rendimiento_modelo.py` |
| Importancia features | `reports/tables/tabular_feature_importance_best_model.csv` | `pages/08_interpretabilidad.py` |
| Matriz confusión | `reports/figures/tabular_best_model_confusion_matrix_absolute.png` | `pages/06_matriz_confusion.py` |
| Casos demo CSV | `demo/demo_cases_binary.csv` | `pages/07_predicciones.py` |
| Parquets por caso | `demo/case_features/case_*.parquet` | `utils/case_eval.py` |
| Fragmentos .npy | `demo/npy_cases/case_*.npy` | `pages/07_predicciones.py` |
| Métricas por caso | `reports/tables/binary_case_level_metrics.csv` | `pages/07_predicciones.py` |

**Comportamiento para archivos .npy sin anotaciones (página 07):**

La app tiene dos modos:
- **Modo A**: archivo .npy disponible + features tabulares asociadas por `case_id` → muestra ECG + predicción binaria.
- **Modo B**: sin .npy → muestra evaluación tabular pre-computada (métricas del CSV).

Si se sube un archivo .npy externo (sin `case_id` conocido), la app **solo muestra la visualización ECG**, no puede predecir. Esto está documentado explícitamente en la app (líneas 543–554 de `07_predicciones.py`):

> "El modelo actual no predice únicamente a partir del archivo .npy, porque fue entrenado con features tabulares derivadas de anotaciones, intervalos RR y metadata clínica."

**Limitación actual:** No existe implementación que convierta un archivo .npy externo en las features tabulares esperadas por el modelo. Una versión futura debería:
1. Detectar picos R en la señal .npy para extraer `rr_prev`, `rr_next`, `hr_inst_from_rr_prev`, `position_in_case`.
2. Solicitar al usuario los metadatos clínicos faltantes.
3. Ejecutar el modelo sobre esas features.

---

## 4. Errores, limitaciones y decisiones importantes

### Fase multiclase descartada
El F1-macro multiclase máximo fue 0.344 (LinearSVC). Clases como AFL, VT, BB tienen muy pocas instancias en el dataset. El F1-macro pondera todas las clases por igual y penaliza fuertemente las raras. El enfoque binario Normal/Anormal tiene más sentido clínico y estadístico.

### Desbalance de clases binario
En el dataset completo: ~78,911 normales vs ~50,262 anormales (en el test). La clase normal tiene más soporte, lo que favorece métricas de accuracy pero puede sesgar el modelo hacia predecir "normal".

### Leakage de `beat_type`
`beat_type` codifica el tipo de latido (N, V, A, etc.) y está directamente correlacionado con `rhythm_label`. Usarla como predictor revelaría la etiqueta al modelo. Está explícitamente bloqueada en `config.TABULAR_LEAKAGE_COLUMNS` y verificada en runtime.

### Leakage de `rhythm_classes`
`rhythm_classes` es una lista de los ritmos del caso completo. Si un latido tuviera esta información como feature, el modelo podría inferir su etiqueta desde ahí. También está bloqueada.

### Separación por `case_id` (crítica)
Dividir por filas individuales sin respetar el `case_id` causaría leakage temporal: el modelo vería latidos del mismo paciente en train y en test. Los latidos del mismo caso comparten metadata y contexto temporal.

### XGBoost y clases ausentes en folds
XGBoost ≥ 2.0 requiere enteros consecutivos en y. Con CV por grupo es frecuente que un fold no tenga alguna clase. El proyecto resuelve esto con `_XGBClassifierSafe` en `src/tabular_search.py` (wrapper con LabelEncoder interno).

### Dependencias de Streamlit
`plotly` es necesario para los gráficos de la app (página 07 usa `plotly.graph_objects`). Verificar que esté en `requirements.txt` (está confirmado). `imbalanced-learn` está en requirements pero no se usa activamente en el flujo binario actual.

### Limitación de archivos .npy externos
La app no puede predecir sobre un .npy externo (señal cruda de un paciente nuevo) porque el modelo requiere features tabulares (metadata clínica + intervalos RR derivados de anotaciones). Este es el límite actual más relevante para el uso real.

### Resultados logreg_balanced no encontrados en el repositorio
Los resultados con sensitivity=0.914, specificity=0.898, ROC-AUC=0.956 para un `logreg_balanced` con `C ≈ 0.0746` y matriz (TN=8319, FP=941, FN=1184, TP=12528) **no se encuentran en ningún archivo del repositorio**. El tamaño del conjunto de test (22,972 registros) es significativamente menor al actual (129,173), lo que sugiere que correspondían a una corrida con `--max-cases` o un subconjunto diferente. El modelo guardado actualmente es LinearSVC con métricas distintas. Antes de documentar esos resultados, habría que re-ejecutar el script con los parámetros exactos y guardar la salida.

### Autoencoder (experimento descartado)
Existe `scripts/05_train_autoencoder_anomaly.py` y `scripts/06_diagnose_autoencoder_pipeline.py`. El autoencoder fue entrenado para detección de anomalías (AUC = 0.45 invertido, ver memory del proyecto). Este experimento es secundario y no se usa en la app actual.

---

## 5. Estructura propuesta para el notebook final

| # Celda | Título | Archivos / funciones a reutilizar |
|---|---|---|
| 1 | Introducción | — |
| 2 | Imports y configuración | `src/config.py` |
| 3 | Rutas del proyecto | `src/config.py` (PROJECT_ROOT, DATA_DIR, MODELS_DIR, etc.) |
| 4 | Carga de datos | `src/data_loading.py`: `load_metadata()`, `load_all_annotations()`, `merge_metadata_and_annotations()` |
| 5 | Exploración inicial (EDA rápido) | Distribución de `rhythm_label`, tamaño del dataset, casos únicos |
| 6 | Limpieza y filtros | `src/preprocessing.py`: `apply_basic_filters()` + función `_drop_label_nans` del script 02 |
| 7 | Features temporales por caso | Función `_add_within_case_time_features` del script 02 |
| 8 | Construcción del target binario | Función `_make_binary_target` del script 03 |
| 9 | Clasificación de features (num/cat) | `src/tabular_search.py`: `classify_features()` |
| 10 | Split por `case_id` | `src/modeling.py`: `make_train_test_group_split_with_coverage()` |
| 11 | Verificación del split (no leakage) | `assert set(groups_train).isdisjoint(set(groups_test))` |
| 12 | Preprocesador | `src/preprocessing.py`: `build_tabular_preprocessor()` |
| 13 | Pipeline + entrenamiento (logreg o LinearSVC) | `src/tabular_search.py`: `build_pipeline_for_model()`, `run_search_for_model()` |
| 14 | Evaluación binaria en test | `src/evaluation.py`: `per_class_report()`, `confusion_matrix_with_totals()` + función `_save_binary_extra_metrics` del script 03 |
| 15 | Métricas globales (F1, ROC-AUC, AP) | `sklearn.metrics`: `roc_auc_score`, `average_precision_score` |
| 16 | Visualización: matriz de confusión | Función `_plot_confusion_matrix` del script 03 |
| 17 | Importancia de features | Función `_extract_feature_importance_binary_safe` del script 03 |
| 18 | Guardado de artefactos | `joblib.dump()` + `json.dump()` + CSVs |
| 19 | Verificación de artefactos para Streamlit | Copiar con `scripts/04_prepare_streamlit_artifacts.py` |
| 20 | Conexión con Streamlit (explicación) | `frontend/app/pages/07_predicciones.py`, `frontend/app/utils/loaders.py` |
| 21 | Conclusiones y limitaciones | — |

---

## 6. Checklist final para mis compañeros

```
[ ] Confirmar que data/raw/physionet_annotations/ tiene metadata.csv
    y los 494 archivos Annotation_file_*.csv

[ ] Ejecutar script 01 (auditoría): python scripts/01_audit_filtered_tabular_dataset.py
    Verificar que no hay errores críticos en los CSVs de salida

[ ] Ejecutar script 02 (dataset): python scripts/02_build_filtered_tabular_modeling_dataset.py
    Verificar que se generó data/processed/filtered_tabular_modeling_dataset.parquet
    Verificar shape esperada (~640 k filas, 482 casos)

[ ] Confirmar columna target binaria: rhythm_label == "N" → "normal", resto → "abnormal"
    (función _make_binary_target en script 03)

[ ] Confirmar exclusión de beat_type como feature (está en TABULAR_LEAKAGE_COLUMNS)

[ ] Confirmar split por case_id, no por fila
    (make_train_test_group_split_with_coverage, test_size=0.20)

[ ] Ejecutar script 03 (entrenamiento binario):
    python scripts/03_run_tabular_binary_hyperparameter_search_FIXED.py
    Verificar que se generaron:
      - models/tabular_best_model_pipeline.joblib
      - models/tabular_best_model_metadata.json
      - reports/tables/tabular_binary_metrics.csv
      - reports/tables/tabular_best_model_classification_report.csv
      - reports/figures/tabular_best_model_confusion_matrix_absolute.png

[ ] Verificar modelo final guardado: leer models/tabular_best_model_metadata.json
    Confirmar winner_model, best_params y winner_metrics

[ ] PENDIENTE: Los resultados logreg_balanced (sensitivity=0.914, ROC-AUC=0.956)
    no están en ningún archivo del repo. Si son los resultados que se quieren
    mostrar, re-ejecutar el script con --models logreg y verificar:
      python scripts/03_run_tabular_binary_hyperparameter_search_FIXED.py --models logreg
    Guardar los resultados explícitamente antes de continuar.

[ ] Ejecutar script 04 (artefactos Streamlit):
    python scripts/04_prepare_streamlit_artifacts.py
    Verificar que frontend/app/app_artifacts/models/ tiene .joblib y .json

[ ] Ejecutar script 05 (casos demo):
    python scripts/05_select_binary_demo_cases.py
    Verificar binary_demo_case_candidates.csv

[ ] Verificar requirements.txt (confirmar plotly, streamlit, scikit-learn, joblib, xgboost)

[ ] Confirmar que la app corre localmente:
    cd frontend/app && streamlit run app.py
    Verificar que la página 07_predicciones.py carga el modelo y muestra los casos demo

[ ] Confirmar qué archivos deben subirse a GitHub:
    - frontend/app/app_artifacts/ → SÍ (modelo, reportes, demo)
    - data/processed/*.parquet    → NO (.gitignore)
    - models/*.joblib             → NO (.gitignore)
    - reports/tables/*.csv        → NO (.gitignore)
    - reports/figures/*.png       → NO (.gitignore)

[ ] Si se quieren añadir ROC-AUC y Average Precision a los reportes,
    modificar _save_binary_extra_metrics en el script 03 para incluirlas
    (sklearn.metrics.roc_auc_score y average_precision_score)
```
