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


class ReservaPrueba:
    """Mantiene la prueba reservada hasta su apertura única en evaluación."""

    def __init__(self, datos, objetivo, ids_entrenamiento):
        if not datos.index.equals(objetivo.index):
            raise ValueError("Las filas y etiquetas de prueba están desalineadas.")
        if not datos.index.is_unique or not objetivo.index.is_unique:
            raise ValueError("Los identificadores de prueba deben ser únicos.")
        if not set(datos.index).isdisjoint(ids_entrenamiento):
            raise ValueError("Entrenamiento y prueba comparten identificadores de origen.")
        self.n_registros = len(datos)
        self.__datos = datos.copy(deep=True)
        self.__objetivo = objetivo.copy(deep=True)
        self.__abierta = False

    def abrir(self):
        if self.__abierta:
            raise RuntimeError("La reserva de prueba ya se abrió para evaluación.")
        datos, objetivo = self.__datos, self.__objetivo
        self.__datos = self.__objetivo = None
        self.__abierta = True
        return datos, objetivo


# Bloque 1: preprocesamiento de datos


def preparar_datos():
    """Devuelve el entrenamiento y mantiene la prueba encapsulada."""
    titulo_bloque("BLOQUE 1: PREPROCESAMIENTO DE DATOS")
    # Ingesta y binarización según UCI [1].
    dataset = fetch_ucirepo(id=45)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    predictoras = [c for c in df.columns if c != VARIABLE_ORIGINAL]
    registros_ingesta = len(df)
    df[VARIABLE_OBJETIVO] = (df[VARIABLE_ORIGINAL] > 0).astype(int)

    # Exclusión de registros incompletos [15].
    df = df.dropna()
    registros_completos = len(df)
    X, y = df[predictoras], df[VARIABLE_OBJETIVO]
    # Partición estratificada antes del balanceo [7].
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, random_state=SEMILLA, stratify=y,
    )
    if not X_train.index.equals(y_train.index) or not X_train.index.is_unique:
        raise ValueError("Las filas y etiquetas de entrenamiento están desalineadas.")
    reserva = ReservaPrueba(X_test, y_test, X_train.index)
    del X_test, y_test
    X_real, y_real = X_train.copy(), y_train.copy()


    niveles_entrenamiento = df.loc[X_real.index, VARIABLE_ORIGINAL].value_counts().sort_index()
    clases_previas = y_real.value_counts().sort_index()
    titulo_subbloque("1.1. Ingesta y delimitación de conjuntos")
    print(f"Registros obtenidos: {registros_ingesta}; completos: {registros_completos}")
    print(f"Predictoras: {len(predictoras)}; entrenamiento real: {len(X_real)}")
    print(f"Reserva encapsulada: {reserva.n_registros} registros; se abre en evaluación")

    titulo_subbloque("1.2. Binarización de la variable objetivo")

    print("Niveles de num SOLO en entrenamiento: " + ", ".join(
        f"{nivel}: {cantidad}" for nivel, cantidad in niveles_entrenamiento.items()))
    print(f"Ausencia: {clases_previas[0]}; presencia: {clases_previas[1]}")
    print(f"Prevalencia del entrenamiento real: {y_real.mean():.4f}")

    titulo_subbloque("1.3. Exclusión de registros incompletos")

    print(f"Registros excluidos por ausentes: {registros_ingesta - registros_completos}")
    print("Ceros observados conservados; sin imputación ni revisión de duplicados.")

    titulo_subbloque("1.4. Composición del conjunto de predictoras")

    print(f"Predictoras conservadas: {', '.join(predictoras)}")
    print(f"Categóricas: {', '.join(CATEGORICAS)}; conteo discreto: ca")
    print(f"Continuas: {', '.join(CONTINUAS)}")
    print("Valores atípicos conservados; sin recorte ni eliminación por IQR.")

    titulo_subbloque("1.5. Partición y frecuencias del entrenamiento")

    print(f"Partición estratificada por diagnóstico: 80/20; semilla {SEMILLA}")
    print("Valores observados SOLO en entrenamiento real (valor: frecuencia):")
    for columna in NO_INTERPOLABLES:
        frecuencias = X_real[columna].value_counts().sort_index()
        print(f"    {columna}: " + ", ".join(
            f"{int(valor)}: {int(n)}" for valor, n in frecuencias.items()))
    print("Este resumen no comprueba categorías ni prevalencia de prueba.")

    del dataset, df, X, y

    titulo_subbloque("1.6. Balanceo preliminar del entrenamiento")

    # SMOTENC aplicado exclusivamente al entrenamiento [9], [10].
    indices = [X_real.columns.get_loc(c) for c in NO_INTERPOLABLES]
    X_balanceado, y_balanceado = SMOTENC(
        categorical_features=indices, random_state=SEMILLA,
    ).fit_resample(X_real, y_real)
    clases_posteriores = y_balanceado.value_counts().sort_index()
    print(f"Antes: {clases_previas[0]} ausencia / {clases_previas[1]} presencia")
    print(f"Después: {clases_posteriores[0]} ausencia / {clases_posteriores[1]} presencia")
    print(f"Sintéticos añadidos: {len(X_balanceado) - len(X_real)}")
    print("Esta salida no alimenta CV ni se incluye como referencia del artefacto.")
    return X_real, y_real, predictoras, reserva

(X_train_sin_balancear, y_train_sin_balancear,
 predictoras, reserva_prueba) = preparar_datos()


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
print(f"Reserva sin abrir: {reserva_prueba.n_registros} registros (20 %)")
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
    "n_prueba": reserva_prueba.n_registros,
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
    "n_prueba": reserva_prueba.n_registros,
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
