#!/usr/bin/env bash
# A2O2: despliegue local de Ollama para la prueba de concepto.
set -euo pipefail

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
PROMPT_FILE="$(dirname "$0")/Artefacto_A1O2.md"

green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$1"; }
red()   { printf '\033[0;31m%s\033[0m\n' "$1"; }

# 1. Instalación de Ollama
echo "== [1/4] Verificando instalación de Ollama =="
if command -v ollama >/dev/null 2>&1; then
    green "Ollama ya está instalado: $(ollama --version 2>/dev/null || echo 'ok')"
else
    yellow "Ollama no encontrado. Instalando..."
    case "$(uname -s)" in
        Linux*)  curl -fsSL https://ollama.com/install.sh | sh ;;
        Darwin*) red "En macOS descargue el instalador desde https://ollama.com/download"; exit 1 ;;
        *)       red "SO no soportado por este script. Instale Ollama manualmente: https://ollama.com/download"; exit 1 ;;
    esac
    green "Ollama instalado."
fi

# 2. Despliegue del servicio
echo "== [2/4] Levantando el servicio de Ollama =="
if curl -fsS "${HOST}/api/tags" >/dev/null 2>&1; then
    green "El servicio ya está activo en ${HOST}."
else
    yellow "Iniciando 'ollama serve' en segundo plano..."
    ollama serve >/tmp/ollama_serve.log 2>&1 &
    for i in $(seq 1 30); do
        sleep 1
        if curl -fsS "${HOST}/api/tags" >/dev/null 2>&1; then break; fi
    done
    if curl -fsS "${HOST}/api/tags" >/dev/null 2>&1; then
        green "Servicio activo en ${HOST}."
    else
        red "No se pudo levantar el servicio. Revise /tmp/ollama_serve.log"; exit 1
    fi
fi

# 3. Descarga del modelo
echo "== [3/4] Descargando el modelo ${MODEL} =="
if ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
    green "El modelo ${MODEL} ya está disponible."
else
    ollama pull "${MODEL}"
    green "Modelo ${MODEL} descargado."
fi

# 4. Prueba de comunicación
echo "== [4/4] Prueba de comunicación con el modelo =="
if [ -f "${PROMPT_FILE}" ]; then
    TEST_PROMPT="Responde en una sola frase confirmando que puedes leer instrucciones clínicas."
else
    TEST_PROMPT="Responde solo: OK"
fi

RESPONSE="$(curl -fsS "${HOST}/api/generate" \
    -d "$(printf '{"model":"%s","prompt":"%s","stream":false}' "${MODEL}" "${TEST_PROMPT}")" \
    | sed -n 's/.*"response":"\([^"]*\)".*/\1/p')"

if [ -n "${RESPONSE}" ]; then
    green "Comunicación verificada. Respuesta del modelo:"
    echo "    ${RESPONSE}"
    green "== Servicio local de Ollama listo =="
else
    red "El modelo no devolvió respuesta. Revise el servicio."; exit 1
fi
