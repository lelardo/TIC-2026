"""Calcula contribuciones SHAP con la preparación de la prueba de concepto."""

import logging

import numpy as np
import shap

from .bundle import ModelBundle

logger = logging.getLogger(__name__)


class ShapExplainer:
    """Inicializa el explainer adecuado y calcula SHAP por caso."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self._explainer = None

    def init(self) -> None:
        """Construye el explainer. Debe llamarse una vez al arrancar."""
        b = self.bundle
        if b.is_xgboost:
            self._explainer = shap.TreeExplainer(
                b.model, feature_perturbation="tree_path_dependent"
            )
            logger.info("TreeExplainer (tree_path_dependent) listo.")
        else:
            x_bg = b.x_train_ref.copy()
            for col in b.log_features or []:
                if col in x_bg.columns:
                    x_bg[col] = np.log1p(x_bg[col].astype(float))
            x_bg = b.scaler.transform(x_bg.values)
            x_bg = b.pca.transform(x_bg)
            self._explainer = shap.LinearExplainer(b.model, x_bg)
            logger.info("LinearExplainer (PCA background) listo.")

    def shap_for(self, x_input) -> tuple[np.ndarray, float]:
        """Devuelve (shap_vals en espacio de features, base_value)."""
        if self._explainer is None:
            raise RuntimeError("Explainer no inicializado; llama a init() primero.")

        b = self.bundle
        expl = self._explainer

        if b.is_xgboost:
            raw = expl.shap_values(x_input)

            if isinstance(raw, list):
                shap_vals = np.array(
                    raw[1][0] if len(raw) > 1 else raw[0][0]
                ).flatten()
            elif len(np.array(raw).shape) == 3:
                shap_vals = np.array(raw)[0, :, 1].flatten()
            else:
                shap_vals = np.array(raw[0]).flatten()

            base_value = float(
                expl.expected_value[1]
                if isinstance(expl.expected_value, (list, np.ndarray))
                and len(np.atleast_1d(expl.expected_value)) > 1
                else expl.expected_value
            )
        else:
            sv_pca = expl.shap_values(x_input)
            sv_pca_clean = (
                sv_pca[0].flatten() if isinstance(sv_pca, list)
                else np.array(sv_pca).flatten()
            )
            shap_vals = (sv_pca_clean @ b.pca.components_).flatten()
            base_value = float(
                expl.expected_value[0]
                if isinstance(expl.expected_value, (list, np.ndarray))
                else expl.expected_value
            )

        return shap_vals, base_value
