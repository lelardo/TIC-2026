"""A3O1: preprocesamiento y entrenamiento de LR y XGBoost. Fuentes en REFERENCIAS.md."""

# Dependencias
from importlib.metadata import version

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as PipelineConRemuestreo
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, TunedThresholdClassifierCV, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

SEMILLA = 42
PROPORCION_PRUEBA = 0.20

VARIABLE_ORIGINAL = "num"
VARIABLE_OBJETIVO = "objetivo"

CATEGORICAS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
CONTEOS_DISCRETOS = ["ca"]
CONTINUAS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# ca conserva valores enteros durante el remuestreo.
NO_INTERPOLABLES = CATEGORICAS + CONTEOS_DISCRETOS

ANCHO = 78


def titulo_bloque(texto):
    """Imprime el encabezado de un bloque."""
    print()
    print("=" * ANCHO)
    print(texto)
    print("=" * ANCHO)


def titulo_subbloque(texto):
    """Imprime el encabezado de un sub-bloque."""
    print()
    print(texto)
    print("-" * ANCHO)


# Bloque 1: preprocesamiento de datos

titulo_bloque("BLOQUE 1: PREPROCESAMIENTO DE DATOS")

# 1.1. Ingesta
# Obtiene Cleveland de UCI [1].

dataset = fetch_ucirepo(id=45)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

predictoras = [columna for columna in df.columns if columna != VARIABLE_ORIGINAL]
registros_ingesta = len(df)

titulo_subbloque("1.1. Ingesta")
print(f"Registros obtenidos:   {registros_ingesta}")
print(f"Variables predictoras: {len(predictoras)}")
print(f"Variable objetivo:     {VARIABLE_ORIGINAL} "
      f"(niveles {sorted(df[VARIABLE_ORIGINAL].unique().tolist())})")

# 1.2. Binarización de la variable objetivo
# Agrupa num > 0 como presencia de enfermedad [1].

niveles_originales = df[VARIABLE_ORIGINAL].value_counts().sort_index()

df[VARIABLE_OBJETIVO] = (df[VARIABLE_ORIGINAL] > 0).astype(int)

clases = df[VARIABLE_OBJETIVO].value_counts().sort_index()

titulo_subbloque("1.2. Binarización de la variable objetivo")
print("Niveles de origen:  "
      + ",  ".join(f"{nivel}: {cantidad}"
                   for nivel, cantidad in niveles_originales.items()))
print(f"Ausencia (0):       {clases[0]} registros "
      f"({clases[0] / registros_ingesta * 100:.2f} %)")
print(f"Presencia (1):      {clases[1]} registros "
      f"({clases[1] / registros_ingesta * 100:.2f} %)")
print(f"Razón entre clases: {clases[0] / clases[1]:.2f} a 1")

# 1.3. Eliminación de registros incompletos
# Excluye registros incompletos sin imputación [15].

ausentes = df.isnull().sum()
ausentes = ausentes[ausentes > 0]
prevalencia_previa = df[VARIABLE_OBJETIVO].mean()

df = df.dropna().reset_index(drop=True)

titulo_subbloque("1.3. Eliminación de registros incompletos")
print("Valores ausentes:      "
      + ",  ".join(f"{variable}: {cantidad}"
                   for variable, cantidad in ausentes.items()))
print(f"Registros eliminados:  {registros_ingesta - len(df)}")
print(f"Registros conservados: {len(df)} de {registros_ingesta} "
      f"({len(df) / registros_ingesta * 100:.2f} %)")
print(f"Prevalencia:           {prevalencia_previa:.4f} -> "
      f"{df[VARIABLE_OBJETIVO].mean():.4f} "
      f"({(df[VARIABLE_OBJETIVO].mean() - prevalencia_previa) * 100:+.2f} "
      f"puntos porcentuales)")

# 1.4. Composición del conjunto de predictoras
# Conserva las trece predictoras y los atípicos según el EDA.

titulo_subbloque("1.4. Composición del conjunto de predictoras")
print(f"Predictoras conservadas: {len(predictoras)}; ninguna se descarta.")
print(f"    categóricas ({len(CATEGORICAS)}): {', '.join(CATEGORICAS)}")
print(f"    conteos     ({len(CONTEOS_DISCRETOS)}): {', '.join(CONTEOS_DISCRETOS)}")
print(f"    continuas   ({len(CONTINUAS)}): {', '.join(CONTINUAS)}")
print("Valores atípicos: se conservan; no se aplica eliminación ni recorte.")

# 1.5. Partición en entrenamiento y prueba
# Divide 80/20 por diagnóstico antes del balanceo [7].

X = df[predictoras]
y = df[VARIABLE_OBJETIVO]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=PROPORCION_PRUEBA,
    random_state=SEMILLA,
    stratify=y,
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

valores_sin_representacion = [
    f"{columna}={int(categoria)}"
    for columna in NO_INTERPOLABLES
    for categoria in sorted(df[columna].unique())
    if int(X_test[columna].value_counts().get(categoria, 0)) == 0
]

titulo_subbloque("1.5. Partición en entrenamiento y prueba")
print(f"Entrenamiento: {len(X_train)} registros "
      f"({len(X_train) / len(df) * 100:.0f} %), prevalencia {y_train.mean():.4f}")
print(f"Prueba:        {len(X_test)} registros "
      f"({len(X_test) / len(df) * 100:.0f} %), prevalencia {y_test.mean():.4f}")
if valores_sin_representacion:
    print("Valores discretos sin representación en prueba: "
          + ", ".join(valores_sin_representacion))
else:
    print("Todos los valores categóricos y del conteo mantienen representación en prueba.")

# 1.6. Balanceo de la clase minoritaria
# Aplica SMOTENC al entrenamiento y conserva su copia real [9], [10].

# La validación parte del entrenamiento real, antes de SMOTENC.

X_train_sin_balancear = X_train.copy()
y_train_sin_balancear = y_train.copy()

clases_previas = y_train.value_counts().sort_index()

indices_no_interpolables = [X_train.columns.get_loc(columna) for columna in NO_INTERPOLABLES]

X_train, y_train = SMOTENC(
    categorical_features=indices_no_interpolables,
    random_state=SEMILLA,
).fit_resample(X_train, y_train)

clases_posteriores = y_train.value_counts().sort_index()
sinteticos = int(clases_posteriores[1] - clases_previas[1])

titulo_subbloque("1.6. Balanceo de la clase minoritaria")
print(f"Entrenamiento antes:   {clases_previas[0]} ausencia / "
      f"{clases_previas[1]} presencia "
      f"(razón {clases_previas[0] / clases_previas[1]:.2f} a 1)")
print(f"Entrenamiento después: {clases_posteriores[0]} ausencia / "
      f"{clases_posteriores[1]} presencia "
      f"(razón {clases_posteriores[0] / clases_posteriores[1]:.2f} a 1)")
print(f"Muestras sintéticas generadas: {sinteticos}")
print(f"Total de entrenamiento: {len(X_train)} registros "
      f"({len(X_train) - sinteticos} reales, {sinteticos} sintéticos)")
print(f"Conjunto de prueba:     {len(X_test)} registros, sin modificar "
      f"(prevalencia {y_test.mean():.4f})")


# Bloque 2: entrenamiento de los modelos

titulo_bloque("BLOQUE 2: ENTRENAMIENTO DE LOS MODELOS")

PLIEGUES = 5
NOMINALES_LR = ["cp", "restecg", "thal"]
RESTO_LR = [c for c in predictoras if c not in NOMINALES_LR]
UMBRALES = np.linspace(0.30, 0.70, 41)
CRITERIO_UMBRAL = "recall"
UMBRAL_XGB = 0.50
HIPERPARAMETROS_XGB = {
    "max_depth": 5,
    "learning_rate": 0.30,
    "n_estimators": 50,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "min_child_weight": 1.0,
}

# 2.1. Protocolo de ajuste
# Construye los pliegues con registros reales; remuestrea en cada ajuste [16].

particiones_internas = list(StratifiedKFold(
    n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA,
).split(X_train_sin_balancear, y_train_sin_balancear))

indices_no_interpolables_cv = [
    X_train_sin_balancear.columns.get_loc(c) for c in NO_INTERPOLABLES
]

titulo_subbloque("2.1. Protocolo de ajuste")
print(f"Entrenamiento real: {len(X_train_sin_balancear)} registros (80 %)")
print(f"Prueba disponible:  {len(X_test)} registros (20 %); no se evalúa aquí")
print(f"Selección interna:  {PLIEGUES} pliegues estratificados, semilla {SEMILLA}")
print("SMOTENC se ajusta dentro de cada entrenamiento, antes de preparar LR.")
print("XGBoost recibe las variables originales después del remuestreo.")

# 2.2. Preparación específica de LR
# Codifica cp, restecg y thal; estandariza las demás columnas [7], [17], [18].

preparacion_lr = ColumnTransformer([
    ("nominales", OneHotEncoder(
        drop="first", sparse_output=False,
        handle_unknown="infrequent_if_exist", min_frequency=1,
    ), NOMINALES_LR),
    ("resto", StandardScaler(), RESTO_LR),
])

titulo_subbloque("2.2. Preparación específica de LR")
print(f"One-hot:        {', '.join(NOMINALES_LR)}")
print(f"Estandarizadas: {', '.join(RESTO_LR)}")

# 2.3. Configuración fija de LR
# Configuración elegida experimentalmente; antecedentes de LR [19]–[21].

procedimiento_lr = PipelineConRemuestreo([
    ("balanceo", SMOTENC(
        categorical_features=indices_no_interpolables_cv, random_state=SEMILLA,
    )),
    ("preparacion", preparacion_lr),
    ("modelo", LogisticRegression(
        solver="liblinear", l1_ratio=1.0, C=1.0,
        max_iter=100, tol=1e-4, class_weight=None, random_state=SEMILLA,
    )),
])

titulo_subbloque("2.3. Configuración fija de LR")
print("Penalización L1 | solver liblinear | C=1 | class_weight=None")
print("max_iter=100 | tol=0.0001 | semilla=42")

# 2.4. Selección de umbral y reajuste final
# Selecciona el umbral por Recall mediante validación interna [22], [23].

modelo_lineal = TunedThresholdClassifierCV(
    procedimiento_lr,
    scoring=CRITERIO_UMBRAL,
    thresholds=UMBRALES,
    cv=particiones_internas,
    refit=True,
    response_method="predict_proba",
    n_jobs=1,
    store_cv_results=True,
)
modelo_lineal.fit(X_train_sin_balancear, y_train_sin_balancear)

umbral_lineal = float(modelo_lineal.best_threshold_)
clasificador_lineal = modelo_lineal.estimator_
curva_umbral_lr = pd.DataFrame({
    "umbral": modelo_lineal.cv_results_["thresholds"],
    "recall_medio_interno": modelo_lineal.cv_results_["scores"],
})

configuracion_entrenamiento_lr = {
    "modelo": "Regresión Logística",
    "semilla": SEMILLA,
    "proporcion_prueba": PROPORCION_PRUEBA,
    "n_entrenamiento_real": len(X_train_sin_balancear),
    "n_prueba": len(X_test),
    "columnas": list(X_train_sin_balancear.columns),
    "nominales_one_hot": NOMINALES_LR,
    "columnas_estandarizadas": RESTO_LR,
    "no_interpolables_smotenc": NO_INTERPOLABLES,
    "hiperparametros": clasificador_lineal.named_steps["modelo"].get_params(),
    "seleccion_umbral": {
        "metodo": "TunedThresholdClassifierCV",
        "criterio": CRITERIO_UMBRAL,
        "candidatos": UMBRALES.tolist(),
        "pliegues": PLIEGUES,
        "particiones": [
            {"ajuste_posiciones": a.tolist(), "validacion_posiciones": v.tolist()}
            for a, v in particiones_internas
        ],
        "umbral_seleccionado": umbral_lineal,
        "score_interno_seleccion": float(modelo_lineal.best_score_),
    },
    "versiones": {nombre: version(nombre) for nombre in (
        "numpy", "pandas", "scikit-learn", "imbalanced-learn", "ucimlrepo", "xgboost",
    )},
    "prueba_evaluada_en_este_archivo": False,
    "historial_prueba": "La misma partición fue consultada durante la experimentación",
}

titulo_subbloque("2.4. Selección de umbral y reajuste final")
print(f"Umbral seleccionado: {umbral_lineal:.4f}")
print(f"Recall interno de selección: {modelo_lineal.best_score_:.6f}")
print("El score anterior es de selección interna; no es rendimiento en prueba.")
print(f"Reajuste final: {len(X_train_sin_balancear)} registros reales, con SMOTENC interno")
print("Disponibles: modelo_lineal, clasificador_lineal y umbral_lineal.")

# 2.5. Configuración y ajuste de XGBoost
# Configuración experimental fija; descripción de parámetros en [25].

modelo_ensamble = PipelineConRemuestreo([
    ("balanceo", SMOTENC(
        categorical_features=indices_no_interpolables_cv, random_state=SEMILLA,
    )),
    ("modelo", XGBClassifier(random_state=SEMILLA, **HIPERPARAMETROS_XGB)),
])
modelo_ensamble.fit(X_train_sin_balancear, y_train_sin_balancear)
clasificador_ensamble = modelo_ensamble
umbral_ensamble = UMBRAL_XGB

configuracion_entrenamiento_xgb = {
    "modelo": "XGBoost",
    "semilla": SEMILLA,
    "proporcion_prueba": PROPORCION_PRUEBA,
    "n_entrenamiento_real": len(X_train_sin_balancear),
    "n_prueba": len(X_test),
    "columnas": list(X_train_sin_balancear.columns),
    "no_interpolables_smotenc": NO_INTERPOLABLES,
    "preparacion_especifica": "Sin escalado ni one-hot adicional",
    "hiperparametros": modelo_ensamble.named_steps["modelo"].get_params(),
    "umbral": {"metodo": "Corte fijo", "valor": umbral_ensamble},
    "versiones": configuracion_entrenamiento_lr["versiones"].copy(),
    "prueba_evaluada_en_este_archivo": False,
    "historial_prueba": configuracion_entrenamiento_lr["historial_prueba"],
}

titulo_subbloque("2.5. Configuración y ajuste de XGBoost")
for parametro, valor in HIPERPARAMETROS_XGB.items():
    print(f"    {parametro}: {valor}")
print(f"Umbral fijo: {umbral_ensamble:.2f}; sin búsqueda de umbral para XGBoost")
print(f"Ajuste: {len(X_train_sin_balancear)} registros reales, con SMOTENC interno")
print("Ambos modelos quedan disponibles para la actividad de evaluación.")
