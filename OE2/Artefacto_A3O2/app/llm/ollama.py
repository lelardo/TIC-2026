"""Cliente HTTP del servicio local de Ollama."""

import json
import logging

import httpx

from ..config import (
    OLLAMA_API_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_GENERATE_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_WARMUP_TIMEOUT,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, api_url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL):
        self.api_url = api_url
        self.model = model

    async def health(self) -> bool:
        """True si Ollama responde a un prompt mínimo."""
        try:
            timeout = httpx.Timeout(OLLAMA_WARMUP_TIMEOUT, connect=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": "OK",
                        "stream": False,
                        "keep_alive": OLLAMA_KEEP_ALIVE,
                    },
                )
                resp.raise_for_status()
                return True
        except Exception as exc:
            logger.warning("Ollama no disponible: %s", exc)
            return False

    async def warmup(self) -> None:
        """Despierta el modelo al arrancar; no lanza excepción si falla."""
        ok = await self.health()
        if ok:
            logger.info("Ollama activo (%s).", self.model)
        else:
            logger.warning(
                "El servidor opera sin LLM hasta que Ollama este activo (%s).",
                self.model,
            )

    async def generate(self, prompt: str) -> str:
        """Generación completa (no streaming)."""
        timeout = httpx.Timeout(OLLAMA_GENERATE_TIMEOUT, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def stream(self, prompt: str):
        """Generación en streaming; produce fragmentos de texto a medida que llegan."""
        timeout = httpx.Timeout(OLLAMA_GENERATE_TIMEOUT, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = chunk.get("response", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
