"""A2O1: preprocesamiento común de los datos. Fuentes en REFERENCIAS.md."""

# Dependencias
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

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

    titulo_subbloque("1.6. Balanceo del entrenamiento")

    # SMOTENC aplicado exclusivamente al entrenamiento [9], [10].
    indices = [X_real.columns.get_loc(c) for c in NO_INTERPOLABLES]
    X_balanceado, y_balanceado = SMOTENC(
        categorical_features=indices, random_state=SEMILLA,
    ).fit_resample(X_real, y_real)
    clases_posteriores = y_balanceado.value_counts().sort_index()
    print(f"Antes: {clases_previas[0]} ausencia / {clases_previas[1]} presencia")
    print(f"Después: {clases_posteriores[0]} ausencia / {clases_posteriores[1]} presencia")
    print(f"Sintéticos añadidos: {len(X_balanceado) - len(X_real)}")
    print("La validación posterior parte de la copia real sin balancear.")
    return X_real, y_real, X_balanceado, y_balanceado, predictoras, reserva

(X_train_sin_balancear, y_train_sin_balancear, X_train, y_train,
 predictoras, reserva_prueba) = preparar_datos()
