"""Rutas, servicio de lenguaje y metadatos de las variables clínicas."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

_BUNDLE_CANDIDATES = [
    BASE_DIR / "best_model_bundle.joblib",
    BASE_DIR.parent / "best_model_bundle.joblib",
]


def _resolve_bundle() -> Path:
    env = os.getenv("BUNDLE_PATH")
    if env:
        return Path(env)
    for cand in _BUNDLE_CANDIDATES:
        if cand.exists():
            return cand

    return _BUNDLE_CANDIDATES[0]

BUNDLE_PATH = _resolve_bundle()
PROMPT_PATH = Path(os.getenv("PROMPT_PATH", Path(__file__).resolve().parent / "llm" / "PROMPT.md"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_API_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_WARMUP_TIMEOUT = float(os.getenv("OLLAMA_WARMUP_TIMEOUT", "500"))
OLLAMA_GENERATE_TIMEOUT = float(os.getenv("OLLAMA_GENERATE_TIMEOUT", "500"))

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

FEATURE_META = {
    "cp": {
        "label": "Tipo de dolor de pecho",
        "clinical": "Tipo de dolor de pecho",
        "type": "select",
        "default": 4,
        "options": [
            {"value": 1, "label": "Angina típica"},
            {"value": 2, "label": "Angina atípica"},
            {"value": 3, "label": "Dolor no anginoso"},
            {"value": 4, "label": "Asintomático"},
        ],
    },
    "thalach": {
        "label": "Frecuencia cardíaca máxima",
        "clinical": "Frecuencia cardíaca máxima alcanzada durante el esfuerzo físico",
        "type": "number",
        "default": 130,
        "min": 60,
        "max": 220,
        "step": 1,
        "unit": "lpm",
    },
    "exang": {
        "label": "Angina inducida por ejercicio",
        "clinical": "Angina inducida por ejercicio",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Sí"},
        ],
    },
    "oldpeak": {
        "label": "Depresión del segmento ST",
        "clinical": "Depresión del segmento ST inducida por el ejercicio",
        "type": "number",
        "default": 2.5,
        "min": 0.0,
        "max": 6.5,
        "step": 0.1,
        "unit": "mm",
    },
    "slope": {
        "label": "Pendiente del segmento ST",
        "clinical": "Pendiente del segmento ST en el pico del esfuerzo",
        "type": "select",
        "default": 2,
        "options": [
            {"value": 1, "label": "1 · Ascendente"},
            {"value": 2, "label": "2 · Plana"},
            {"value": 3, "label": "3 · Descendente"},
        ],
    },
    "ca": {
        "label": "Vasos mayores por fluoroscopia",
        "clinical": "Número de vasos mayores coloreados por fluoroscopia",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "0 vasos"},
            {"value": 1, "label": "1 vaso"},
            {"value": 2, "label": "2 vasos"},
            {"value": 3, "label": "3 vasos"},
        ],
    },
    "thal": {
        "label": "Gammagrafía con talio",
        "clinical": "Centellograma miocárdico con talio",
        "type": "select",
        "default": 7,
        "options": [
            {"value": 3, "label": "3 · Normal"},
            {"value": 6, "label": "6 · Defecto fijo"},
            {"value": 7, "label": "7 · Defecto reversible"},
        ],
    },
}


def feature_meta_for(feature: str) -> dict:
    """Devuelve la metadata de una feature, con un fallback numérico genérico."""
    return FEATURE_META.get(
        feature,
        {
            "label": feature,
            "clinical": feature,
            "type": "number",
            "default": 0,
            "min": 0,
            "max": 1000,
            "step": 0.1,
        },
    )
