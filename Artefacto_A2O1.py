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
