"""A4O1: preparación, entrenamiento, evaluación y exportación de LR. Fuentes en REFERENCIAS.md."""

# Dependencias
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as PipelineConRemuestreo
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, TunedThresholdClassifierCV, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

SEMILLA = 42
PROPORCION_PRUEBA = 0.20
MARGEN_MCC = 0.01  # Margen operativo para el desempate por MCC.

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


def huella_referencia(datos):
    """Identifica valores, columnas, tipos, orden e índices del entrenamiento."""
    firma = sha256()
    esquema = [(str(c), str(t)) for c, t in zip(datos.columns, datos.dtypes)]
    firma.update(json.dumps(esquema, ensure_ascii=False).encode("utf-8"))
    firma.update(pd.util.hash_pandas_object(datos, index=True).to_numpy().tobytes())
    return firma.hexdigest()


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
    """Devuelve entrenamiento real, procedencia y una reserva encapsulada."""
    titulo_bloque("BLOQUE 1: PREPROCESAMIENTO DE DATOS")
    dataset = fetch_ucirepo(id=45)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    predictoras = [c for c in df.columns if c != VARIABLE_ORIGINAL]
    registros_ingesta = len(df)
    df[VARIABLE_OBJETIVO] = (df[VARIABLE_ORIGINAL] > 0).astype(int)

    df = df.dropna()
    registros_completos = len(df)
    X, y = df[predictoras], df[VARIABLE_OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, random_state=SEMILLA, stratify=y,
    )
    if not X_train.index.equals(y_train.index) or not X_train.index.is_unique:
        raise ValueError("Las filas y etiquetas de entrenamiento están desalineadas.")
    reserva = ReservaPrueba(X_test, y_test, X_train.index)
    del X_test, y_test
    X_real, y_real = X_train.copy(), y_train.copy()
    procedencia = {
        "ids_entrenamiento": [int(i) for i in X_real.index],
        "sha256_referencia": huella_referencia(X_real),
        "identificador_origen": "Índice de fila de UCI45; no identificador clínico",
    }

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

    indices = [X_real.columns.get_loc(c) for c in NO_INTERPOLABLES]
    X_balanceado, y_balanceado = SMOTENC(
        categorical_features=indices, random_state=SEMILLA,
    ).fit_resample(X_real, y_real)
    clases_posteriores = y_balanceado.value_counts().sort_index()
    print(f"Antes: {clases_previas[0]} ausencia / {clases_previas[1]} presencia")
    print(f"Después: {clases_posteriores[0]} ausencia / {clases_posteriores[1]} presencia")
    print(f"Sintéticos añadidos: {len(X_balanceado) - len(X_real)}")
    print("Esta salida no alimenta CV ni se incluye como referencia del artefacto.")
    return X_real, y_real, predictoras, procedencia, reserva

(X_train_sin_balancear, y_train_sin_balancear, predictoras,
 procedencia_entrenamiento, reserva_prueba) = preparar_datos()


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
    "procedencia_entrenamiento": procedencia_entrenamiento.copy(),
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
    "procedencia_entrenamiento": procedencia_entrenamiento.copy(),
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


# Bloque 3: evaluación comparativa y selección del modelo
# Calcula Recall, MCC y AUC sobre la reserva, sin nuevos ajustes [26].


def evaluar(modelo, umbral, X_evaluacion, y_evaluacion):
    """Predice una sola vez, sin ajustar el modelo ni modificar su corte."""
    probabilidades = modelo.predict_proba(X_evaluacion)[:, 1]
    prediccion = (probabilidades >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_evaluacion, prediccion, labels=[0, 1],
    ).ravel()
    metricas = {
        "recall": float(recall_score(y_evaluacion, prediccion, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_evaluacion, prediccion)),
        "auc": float(roc_auc_score(y_evaluacion, probabilidades)),
        "vp": int(tp), "fn": int(fn), "fp": int(fp), "vn": int(tn),
    }
    return metricas, probabilidades, prediccion


def seleccionar(resultados):
    """Aplica el criterio fijado, sin buscar configuraciones en prueba."""
    (primer_nombre, primero), (segundo_nombre, segundo) = resultados.items()
    diferencia_vp = primero["vp"] - segundo["vp"]
    if diferencia_vp != 0:
        ganador = primer_nombre if diferencia_vp > 0 else segundo_nombre
        return ganador, "Recall", abs(primero["recall"] - segundo["recall"])
    diferencia_mcc = primero["mcc"] - segundo["mcc"]
    if abs(diferencia_mcc) >= MARGEN_MCC:
        ganador = primer_nombre if diferencia_mcc > 0 else segundo_nombre
        return ganador, "MCC", abs(diferencia_mcc)
    diferencia_auc = primero["auc"] - segundo["auc"]
    if diferencia_auc != 0:
        ganador = primer_nombre if diferencia_auc > 0 else segundo_nombre
        return ganador, "AUC", abs(diferencia_auc)
    return None, "Empate bajo los criterios establecidos", 0.0

titulo_bloque("BLOQUE 3: EVALUACIÓN COMPARATIVA Y SELECCIÓN DEL MODELO")
# Abre prueba únicamente para evaluar los modelos ya fijados.
X_test, y_test = reserva_prueba.abrir()
resultados = {}
predicciones_evaluacion = pd.DataFrame({"objetivo": y_test.to_numpy()})
for nombre, modelo, corte in [
    ("Regresión Logística", modelo_lineal, umbral_lineal),
    ("XGBoost", modelo_ensamble, umbral_ensamble),
]:
    metricas, probabilidades, prediccion = evaluar(modelo, corte, X_test, y_test)
    resultados[nombre] = metricas
    predicciones_evaluacion[nombre + "_probabilidad"] = probabilidades
    predicciones_evaluacion[nombre + "_prediccion"] = prediccion

configuracion_entrenamiento_lr["prueba_evaluada_en_este_archivo"] = True
configuracion_entrenamiento_xgb["prueba_evaluada_en_este_archivo"] = True
configuracion_evaluacion = {
    "n_prueba": len(y_test),
    "n_positivos": int(y_test.sum()),
    "n_negativos": int((1 - y_test).sum()),
    "tipo": "Evaluación de los modelos fijados sobre el 20 % reservado",
    "prioridad": ["recall", "mcc", "auc"],
    "margen_mcc": MARGEN_MCC,
    "umbrales": {"Regresión Logística": umbral_lineal, "XGBoost": umbral_ensamble},
    "reserva_consultada_historicamente": True,
    "artefacto_exportado": False,
}

titulo_subbloque("3.1. Evaluación sobre el conjunto de prueba")
print(f"Registros evaluados: {len(y_test)} "
      f"({int(y_test.sum())} con enfermedad, {int((1 - y_test).sum())} sin enfermedad)")
print("Métricas del 20 % reservado; no son medias de validación cruzada.")
print(f"{'Modelo':<24}{'Recall':>10}{'MCC':>10}{'AUC':>10}")
for nombre, m in resultados.items():
    print(f"{nombre:<24}{m['recall']:>10.4f}{m['mcc']:>10.4f}{m['auc']:>10.4f}")
print()
print("Matriz de confusión:")
print(f"{'Modelo':<24}{'VP':>7}{'FN':>7}{'FP':>7}{'VN':>7}")
for nombre, m in resultados.items():
    print(f"{nombre:<24}{m['vp']:>7}{m['fn']:>7}{m['fp']:>7}{m['vn']:>7}")

modelo_seleccionado, metrica_decisoria, diferencia = seleccionar(resultados)
titulo_subbloque("3.2. Selección del modelo")
print("Prioridad: Recall, después MCC y finalmente AUC")
print(f"Margen operativo de MCC: {MARGEN_MCC:.2f}")
if modelo_seleccionado is None:
    print("Resultado: empate bajo los criterios establecidos")
else:
    print(f"Modelo seleccionado: {modelo_seleccionado}")
    print(f"Métrica decisoria: {metrica_decisoria}; diferencia {diferencia:.4f}")
print("La comparación corresponde a esta partición y configuración.")
print("No se reajustan los modelos ni sus cortes después de evaluar prueba.")


# Bloque 4: exportación de la regresión logística seleccionada
# Exporta LR ajustada con su preparación, umbral y referencia real.


def exportar_lr(ruta, ganador, modelo, configuracion, metricas, X_referencia):
    """Guarda exclusivamente la LR seleccionada, sin ejecutar fit ni búsqueda."""
    if ganador != "Regresión Logística":
        raise ValueError("La exportación está autorizada únicamente para la LR seleccionada.")
    columnas = list(configuracion["columnas"])
    if list(X_referencia.columns) != columnas:
        raise ValueError("El entrenamiento de referencia no coincide con las columnas del modelo.")
    if len(X_referencia) != configuracion["n_entrenamiento_real"]:
        raise ValueError("La referencia debe contener únicamente el entrenamiento real sin balancear.")
    procedencia = configuracion["procedencia_entrenamiento"]
    if X_referencia.index.tolist() != procedencia["ids_entrenamiento"]:
        raise ValueError("Los identificadores u orden de referencia no corresponden al entrenamiento.")
    if huella_referencia(X_referencia) != procedencia["sha256_referencia"]:
        raise ValueError("Los valores de referencia no coinciden con el entrenamiento real registrado.")
    umbral = float(modelo.best_threshold_)
    if umbral != configuracion["seleccion_umbral"]["umbral_seleccionado"]:
        raise ValueError("El umbral del modelo no coincide con el registrado.")
    bundle_lr = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": modelo,
        "model_name": ganador,
        "selected_features": columnas,
        "preprocessor": modelo.estimator_.named_steps["preparacion"],
        "preprocessing_embedded": True,
        "best_threshold": umbral,
        "prediction_rule": "Presencia si predict_proba(X)[:, 1] >= best_threshold",
        "target_classes": {0: "Ausencia", 1: "Presencia"},
        "metrics": {
            "recall": float(metricas["recall"]),
            "mcc": float(metricas["mcc"]),
            "auc_roc": float(metricas["auc"]),
        },
        "confusion_matrix": {k: int(metricas[k]) for k in ("vp", "fn", "fp", "vn")},
        "training_configuration": configuracion,
        "versions": {**configuracion["versiones"], "joblib": version("joblib")},
        "X_train": X_referencia.copy(),  # Referencia real de entrenamiento para SHAP.
    }
    joblib.dump(bundle_lr, ruta)
    return bundle_lr

titulo_bloque("BLOQUE 4: EXPORTACIÓN DE LA REGRESIÓN LOGÍSTICA SELECCIONADA")
if modelo_seleccionado == "Regresión Logística":
    ruta_artefacto_lr = Path(__file__).resolve().parent / "best_model_bundle_lr_aislamiento.joblib"
    bundle_lr = exportar_lr(
        ruta_artefacto_lr, modelo_seleccionado, modelo_lineal,
        configuracion_entrenamiento_lr, resultados["Regresión Logística"],
        X_train_sin_balancear,
    )
    configuracion_evaluacion["artefacto_exportado"] = True
    configuracion_evaluacion["modelo_exportado"] = "Regresión Logística"
    print("Modelo exportado: Regresión Logística")
    print(f"Predictoras originales: {len(bundle_lr['selected_features'])}")
    print(f"Umbral conservado: {bundle_lr['best_threshold']:.4f}")
    print("Preparación ajustada integrada: one-hot y estandarización")
    print(f"Referencia para SHAP: {len(bundle_lr['X_train'])} registros reales de entrenamiento")
    print(f"Artefacto: {ruta_artefacto_lr}")
else:
    print("No se exporta: esta versión requiere que la LR resulte seleccionada.")
