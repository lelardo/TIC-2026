"""Aplicación FastAPI de la prueba de concepto."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import STATIC_DIR
from .llm.ollama import OllamaClient
from .llm.prompt import PromptBuilder
from .ml.bundle import ModelBundle
from .ml.predictor import Predictor
from .schemas import build_patient_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    bundle = ModelBundle()
    predictor = Predictor(bundle)
    predictor.init()

    app.state.predictor = predictor
    app.state.patient_model = build_patient_model(bundle.selected_features)
    app.state.prompt_builder = PromptBuilder()
    app.state.ollama = OllamaClient()

    logger.info("Verificando Ollama...")
    await app.state.ollama.warmup()
    logger.info("Backend listo.")
    yield

app = FastAPI(
    title="API del módulo de explicabilidad",
    description="Predicción, contribuciones SHAP y generación de reportes narrativos.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn

    from .config import HOST, PORT

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
