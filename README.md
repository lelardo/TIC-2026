# Desarrollo de un prototipo de módulo explicable basado en Machine Learning diseñado para la asistencia médica en la predicción de riesgo cardiovascular

Repositorio de los artefactos del Trabajo de Integración Curricular. Contiene
los cuatro entregables completados del primer objetivo y tres pruebas de
concepto del segundo objetivo, pendientes de adaptación y validación.

## Autor

- **Nombre:** Leonardo Aldair Orozco Oviedo.
- **Institución:** Universidad Nacional de Loja.
- **Carrera:** Computación.
- **Itinerario:** Sistemas Inteligentes.
- **Lugar:** Loja, Ecuador.

## Objetivo general

Desarrollar un prototipo de módulo explicable para la predicción de riesgo cardiovascular, basado en el modelo de mejor desempeño entre XGBoost y Regresión Logística según las métricas Recall, MCC y AUC-ROC, interpretado con SHAP y adaptado al lenguaje con Qwen 2.5 Instruct 14B, sobre el dataset UCI Cleveland Heart Disease.

## Objetivo específico 1

Desarrollar un pipeline de entrenamiento y evaluación de los modelos XGBoost y Regresión Logística utilizando el dataset UCI Cleveland Heart Disease, para la selección del modelo de mejor desempeño predictivo según las métricas de Recall, MCC y AUC-ROC.

**Estado:** completado. Sus cuatro actividades alcanzan el 100 %.

| Actividad | Artefacto | Contenido | Estado |
| --- | --- | --- | --- |
| A1O1 | [Artefacto_A1O1.ipynb](Artefacto_A1O1.ipynb) | EDA con resultados, interpretaciones, decisiones y referencias. | Completado · 100 % |
| A2O1 | [Artefacto_A2O1.py](Artefacto_A2O1.py) | Preprocesamiento común, partición estratificada y balanceo del entrenamiento. | Completado · 100 % |
| A3O1 | [Artefacto_A3O1.py](Artefacto_A3O1.py) | Preprocesamiento y entrenamiento de Regresión Logística y XGBoost con las configuraciones seleccionadas. | Completado · 100 % |
| A4O1 | [Artefacto_A4O1.py](Artefacto_A4O1.py) | Pipeline completo: preparación, entrenamiento, evaluación comparativa y exportación de la Regresión Logística seleccionada. | Completado · 100 % |

Los tres scripts son entregables independientes y acumulativos: A3O1 incorpora
el preprocesamiento y A4O1 incorpora también el entrenamiento. La evaluación
compara ambos modelos mediante Recall, MCC y AUC-ROC.

## Objetivo específico 2

Implementar un módulo de explicabilidad basado en SHAP y Qwen 2.5 Instruct 14B sobre el modelo seleccionado, para generar reportes narrativos diseñados para asistir en la interpretación médica de los resultados a través de una interfaz web preliminar.

**Estado:** en desarrollo. Los artefactos disponibles corresponden a pruebas
de concepto y no representan la integración final con la Regresión Logística
seleccionada y Qwen.

| Actividad | Artefacto | Contenido | Estado |
| --- | --- | --- | --- |
| A1O2 | [Artefacto_A1O2.md](Artefacto_A1O2.md) | Prompt preliminar para generar reportes narrativos a partir de la predicción y las contribuciones SHAP. | 80 % · Pendiente de revisión y fundamentación metodológica y clínica. |
| A2O2 | [Artefacto_A2O2.sh](Artefacto_A2O2.sh) | Script de instalación, despliegue y comprobación de comunicación con Ollama. | 80 % · Pendiente de adaptación y validación con Qwen. |
| A3O2 | [Artefacto_A3O2](Artefacto_A3O2/) | Backend FastAPI de predicción, explicabilidad y generación de reportes; incluye su copia del prompt. | 80 % · Pendiente de adaptación al modelo final, revisión de SHAP e integración con Qwen. |
| A4O2 | Sin artefacto publicado. | Desarrollo de la interfaz de usuario e integración del módulo. | 30 % · Alcance definido; la interfaz se construirá desde cero. |

El backend contiene su plantilla en
[app/llm/PROMPT.md](Artefacto_A3O2/app/llm/PROMPT.md). El endpoint
`POST /api/report` la completa con los datos de la predicción y las
contribuciones SHAP para enviarla al servicio local de Ollama.

## Referencias

Las fuentes del objetivo 1 se encuentran en [REFERENCIAS.md](REFERENCIAS.md)
y en la bibliografía del notebook EDA.
