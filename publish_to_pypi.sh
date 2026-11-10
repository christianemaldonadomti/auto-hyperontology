#!/usr/bin/env bash
# ==============================================================================
# Script de Publicación Automática en PyPI para auto-hyperontology
# Lee TWINE_USERNAME y TWINE_PASSWORD desde .env automáticamente
# ==============================================================================

set -e

if [ -f ".env" ]; then
    echo "🔐 Cargando TWINE_PASSWORD desde archivo .env..."
    set -a
    source .env
    set +a
fi

export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"

echo "📦 Limpiando compilaciones anteriores y generando paquetes (.whl y .tar.gz)..."
rm -rf dist/ build/ *.egg-info
python3 -m build

echo "🚀 Subiendo paquete a PyPI..."
python3 -m twine upload dist/*

echo "✅ ¡Publicado exitosamente en https://pypi.org/project/auto-hyperontology/!"
