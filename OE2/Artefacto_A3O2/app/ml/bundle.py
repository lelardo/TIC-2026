"""Carga los componentes del modelo serializado."""

import logging
from pathlib import Path

import joblib

from ..config import BUNDLE_PATH

logger = logging.getLogger(__name__)


class ModelBundle:
    """Contenedor de solo lectura de todo lo necesario para inferir y explicar."""

    def __init__(self, path: Path = BUNDLE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Bundle no encontrado: {path}")

        data = joblib.load(path)

        self.model = data["model"]
        self.model_name: str = data["model_name"]
        self.selected_features: list[str] = list(data["selected_features"])
        self.threshold: float = float(data.get("best_threshold", 0.50))
        self.x_train_ref = data.get("X_train", None)

        preprocessor = data.get("preprocessor", {})
        self.log_features = preprocessor.get("log_features")
        self.scaler = preprocessor.get("scaler")
        self.pca = preprocessor.get("pca")

        self.metrics: dict = data.get("metrics", {})
        self.metrics_xgb: dict = data.get("metrics_xgb", {})
        self.metrics_lr: dict = data.get("metrics_lr", {})

        logger.info("Bundle cargado: %s", self.model_name)
        logger.info("Features: %s", self.selected_features)
        logger.info("Umbral clínico: %.4f", self.threshold)

    @property
    def is_xgboost(self) -> bool:
        return "XGB" in self.model_name or "xgboost" in self.model_name.lower()
