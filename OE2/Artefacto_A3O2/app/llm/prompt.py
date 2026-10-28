"""Construye el prompt narrativo a partir de su plantilla Markdown."""

import logging
from pathlib import Path

from ..config import PROMPT_PATH

logger = logging.getLogger(__name__)


class PromptBuilder:
    def __init__(self, path: Path = PROMPT_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Plantilla de prompt no encontrada: {path}. "
                "Se requiere app/llm/PROMPT.md o una ruta PROMPT_PATH válida."
            )
        self.template = path.read_text(encoding="utf-8")
        logger.info("Plantilla de prompt cargada: %s", path.name)

    def build(self, prediction: dict) -> str:
        prob = prediction["probabilidad"]
        umbral = prediction["umbral_usado"]
        es_riesgo = prediction["es_riesgo"]
        veredicto = "PRESENCIA" if es_riesgo else "AUSENCIA"

        prob_pct = f"{prob:.2f}%"
        umb_pct = f"{umbral:.2f}%"

        lineas = []
        for c in prediction["shap_ranking"]:
            efecto = "POSITIVO (Agrava)" if c["shap_impact"] > 0 else "NEGATIVO (Atenua)"
            lineas.append(
                f"- Atributo: '{c['feature']}' | "
                f"Valor ingresado: {c['value']:.2f} | "
                f"Peso: {c['shap_impact']:.4f} ({efecto})"
            )
        explicacion_shap = "\n".join(lineas)

        if es_riesgo:
            texto_analisis = (
                f"Se determino una sospecha predictiva de PRESENCIA de riesgo coronario "
                f"con una estimacion probabilistica del {prob_pct}, situandose por "
                f"encima del umbral de alerta hospitalaria establecido en {umb_pct}."
            )
        else:
            certeza = f"{100 - prob:.2f}%"
            texto_analisis = (
                f"Se determino la AUSENCIA de criterios de riesgo coronario agudo, "
                f"respaldado por un {certeza} de certeza de normalidad, al ubicarse la "
                f"probabilidad de riesgo ({prob_pct}) holgadamente por debajo del umbral "
                f"de alerta del {umb_pct}."
            )

        return (
            self.template
            .replace("[[VEREDICTO]]", veredicto)
            .replace("[[PROBABILIDAD]]", prob_pct)
            .replace("[[UMBRAL]]", umb_pct)
            .replace("[[EXPLICACION_SHAP]]", explicacion_shap)
            .replace("[[TEXTO_ANALISIS]]", texto_analisis)
        )
