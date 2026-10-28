"""Esquemas Pydantic para las entradas y salidas del backend."""

from typing import Any

from pydantic import BaseModel, Field, create_model

from .config import feature_meta_for


def build_patient_model(selected_features: list[str]):
    """Crea el modelo Pydantic de entrada a partir de las features del bundle."""
    fields: dict[str, Any] = {}
    for feat in selected_features:
        meta = feature_meta_for(feat)
        if meta["type"] == "select":
            values = [o["value"] for o in meta["options"]]
            lo, hi = min(values), max(values)
        else:
            lo, hi = meta.get("min"), meta.get("max")
        fields[feat] = (
            float,
            Field(default=float(meta["default"]), ge=lo, le=hi),
        )
    return create_model("PatientData", **fields)


class ReportRequest(BaseModel):
    """Payload para /api/report: la predicción ya calculada en el cliente."""

    diagnostico: str
    es_riesgo: bool
    probabilidad: float
    umbral_usado: float
    shap_ranking: list[dict]
