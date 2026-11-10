#!/usr/bin/env bash
# ==============================================================================
# Script de Actualización y Sincronización Automática con GitHub
# Autor: Dr. Christian Efraín Maldonado-Sifuentes (christianemaldonadomti)
# ==============================================================================

set -e

COMMIT_MSG="${1:-"feat: upgrade to Proto-AGI v4.0 Forensic (biol1.json schema + index.php Semantic Bridge & ReAct)"}"

echo "========================================================"
echo "🚀 Sincronizando y subiendo cambios a GitHub..."
echo "========================================================"

# 1. Cargar variables desde .env si existe
if [ -f ".env" ]; then
    echo "🔐 Leyendo credenciales desde .env..."
    set -a
    source .env
    set +a
fi

# 2. Inicializar repositorio si no existe .git
if [ ! -d ".git" ]; then
    echo "📦 Inicializando repositorio Git..."
    git init
fi

# 3. Configurar identidad local y recordar credenciales para siempre
git config user.name "Christian Efraín Maldonado-Sifuentes"
git config user.email "christianemaldonadomti@gmail.com"
git config --global credential.helper store

# 4. Proteger archivos de credenciales en .gitignore
if [ ! -f ".gitignore" ] || ! grep -q "^\.env$" .gitignore; then
    cat << 'EOF' >> .gitignore
.env
.pypirc
__pycache__/
*.pyc
dist/
build/
*.egg-info/
EOF
fi

# 5. Asegurar rama main
git branch -M main

# 6. Configurar URL remota
GH_USERNAME="christianemaldonadomti"
if [ -n "$GITHUB_TOKEN" ]; then
    REMOTE_URL="https://${GH_USERNAME}:${GITHUB_TOKEN}@github.com/${GH_USERNAME}/auto-hyperontology.git"
else
    REMOTE_URL="https://github.com/${GH_USERNAME}/auto-hyperontology.git"
fi

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi

# 7. Limpiar remanentes no deseados
rm -f auto_hyperontology/khan_academy.py

# 8. Sincronizar cambios remotos sin romper el trabajo local
echo "🔄 Sincronizando con rama remota..."
git pull origin main --rebase --autostash 2>/dev/null || true

# 9. Agregar todos los archivos
git add -A

# 10. Crear commit si hay diferencias
if git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "ℹ️ No hay cambios pendientes que commitear."
else
    echo "📦 Creando commit: '$COMMIT_MSG'"
    git commit -m "$COMMIT_MSG"
fi

# 11. Empujar a GitHub (con fallback seguro a rebase)
echo "📤 Empujando cambios a origin main..."
if ! git push -u origin main 2>/dev/null; then
    echo "⚠️ Sincronizando diferencias remotas finales..."
    git pull origin main --allow-unrelated-histories --no-rebase -m "merge: sync remote changes" 2>/dev/null || true
    git push -u origin main
fi

# Restaurar URL limpia sin exponer token en 'git remote -v'
if [ -n "$GITHUB_TOKEN" ]; then
    git remote set-url origin "https://github.com/${GH_USERNAME}/auto-hyperontology.git"
fi

echo ""
echo "========================================================"
echo "✅ ¡Repositorio sincronizado y actualizado en GitHub!"
echo "   🔗 https://github.com/christianemaldonadomti/auto-hyperontology"
echo "========================================================"
