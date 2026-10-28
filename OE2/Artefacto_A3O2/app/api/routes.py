"""Endpoints de metadatos, predicción, reporte y estado."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..config import feature_meta_for
from ..schemas import ReportRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

@router.get("/metadata")
async def metadata(request: Request):
    st = request.app.state
    bundle = st.predictor.bundle
    features = [
        {"name": feat, **feature_meta_for(feat)} for feat in bundle.selected_features
    ]
    return {
        "model_name": bundle.model_name,
        "threshold_pct": round(bundle.threshold * 100, 2),
        "metrics": bundle.metrics,
        "metrics_xgb": bundle.metrics_xgb,
        "metrics_lr": bundle.metrics_lr,
        "features": features,
    }

@router.post("/predict")
async def predict(request: Request):
    st = request.app.state
    payload = await request.json()
    try:
        data = st.patient_model(**payload)
        input_dict = data.model_dump()
        return st.predictor.predict(input_dict)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error en /predict")
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/report")
async def report(request: Request, body: ReportRequest):
    st = request.app.state

    if not await st.ollama.health():
        raise HTTPException(
            status_code=503,
            detail="Ollama no está disponible. La predicción y SHAP son válidos; "
            "el asistente narrativo requiere el servicio Llama 3.2 activo.",
        )

    prompt = st.prompt_builder.build(body.model_dump())

    async def token_stream():
        try:
            async for piece in st.ollama.stream(prompt):
                yield piece
        except Exception as exc:
            logger.exception("Error durante el streaming del LLM")
            yield f"\n\n[Error del asistente: {exc}]"

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")

@router.get("/health")
async def health(request: Request):
    st = request.app.state
    return {
        "model": st.predictor.bundle.model_name,
        "model_ready": True,
        "ollama_ready": await st.ollama.health(),
    }
