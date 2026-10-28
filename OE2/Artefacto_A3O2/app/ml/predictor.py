"""Prepara las entradas y combina predicción con contribuciones SHAP."""

import logging

import numpy as np
import pandas as pd

from .bundle import ModelBundle
from .explainer import ShapExplainer

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.explainer = ShapExplainer(bundle)

    def init(self) -> None:
        self.explainer.init()

    def preprocess(self, x_df: pd.DataFrame):
        b = self.bundle
        if b.is_xgboost:
            return x_df.copy()
        x = x_df.copy()
        for col in b.log_features or []:
            x[col] = np.log1p(x[col])
        x = b.scaler.transform(x.values)
        x = b.pca.transform(x)
        return x

    def predict(self, input_dict: dict) -> dict:
        b = self.bundle
        x_caso = pd.DataFrame([input_dict], columns=b.selected_features).astype(float)
        x_input = self.preprocess(x_caso)

        prob = float(b.model.predict_proba(x_input)[0, 1])
        es_riesgo = prob >= b.threshold
        veredicto = (
            "RIESGO DE ENFERMEDAD CORONARIA" if es_riesgo else "SIN RIESGO APARENTE"
        )

        shap_vals, base_value = self.explainer.shap_for(x_input)

        shap_ranking = sorted(
            [
                {
                    "feature": str(feat),
                    "value": float(val),
                    "shap_impact": float(s),
                }
                for feat, val, s in zip(
                    b.selected_features, x_caso.iloc[0].values, shap_vals
                )
            ],
            key=lambda x: abs(x["shap_impact"]),
            reverse=True,
        )

        return {
            "diagnostico": veredicto,
            "es_riesgo": es_riesgo,
            "probabilidad": round(prob * 100, 2),
            "umbral_usado": round(b.threshold * 100, 2),
            "base_value": base_value,
            "shap_ranking": shap_ranking,
            "model_name": b.model_name,
        }
