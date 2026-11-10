#!/usr/bin/env bash
# ==============================================================================
# Script de Actualización Rápida a GitHub para auto-hyperontology
# Autor: Dr. Christian Efraín Maldonado-Sifuentes (christianemaldonadomti)
# ==============================================================================

set -e

# Mensaje de commit personalizado o por defecto
COMMIT_MSG="${1:-"feat: update auto-hyperontology with qwen 125B extraction, live benchmark, and Drive cache"}"

echo "========================================================"
echo "🚀 Subiendo cambios a GitHub (christianemaldonadomti)..."
echo "========================================================"

# Asegurar rama main
git branch -M main 2>/dev/null || true

# Configurar identidad si no está configurada localmente
git config user.name "Christian Efraín Maldonado-Sifuentes" 2>/dev/null || true
git config user.email "christianemaldonadomti@gmail.com" 2>/dev/null || true

# Configurar remote si no existe o actualizarlo
REMOTE_URL="https://github.com/christianemaldonadomti/auto-hyperontology.git"
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi

# Eliminar archivos no deseados (ej. khan_academy.py si existiera)
rm -f auto_hyperontology/khan_academy.py

# Añadir todos los cambios (incluyendo archivos eliminados y notebooks nuevos)
git add -A

# Verificar si hay cambios pendientes de commit
if git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "ℹ️ No hay cambios pendientes que commitear."
else
    echo "📦 Creando commit: '$COMMIT_MSG'"
    git commit -m "$COMMIT_MSG"
fi

# Subir a la rama main en GitHub
echo "📤 Empujando cambios a origin main..."
git push -u origin main

echo ""
echo "========================================================"
echo "✅ ¡Repositorio actualizado exitosamente en GitHub!"
echo "   🔗 https://github.com/christianemaldonadomti/auto-hyperontology"
echo "========================================================"
