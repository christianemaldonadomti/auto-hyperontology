# ==============================================================================
# Script de Actualización Rápida a GitHub para Windows PowerShell
# Autor: Dr. Christian Efraín Maldonado-Sifuentes (christianemaldonadomti)
# ==============================================================================

param(
    [string]$Message = "feat: update auto-hyperontology with qwen 125B extraction, live benchmark, and Drive cache"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "🚀 Subiendo cambios a GitHub (christianemaldonadomti)..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Asegurar rama main
git branch -M main 2>$null

# Configurar identidad
git config user.name "Christian Efraín Maldonado-Sifuentes"
git config user.email "christianemaldonadomti@gmail.com"

# Configurar remote oficial
$RemoteUrl = "https://github.com/christianemaldonadomti/auto-hyperontology.git"
git remote set-url origin $RemoteUrl 2>$null
if ($LASTEXITCODE -ne 0) {
    git remote add origin $RemoteUrl
}

# Eliminar remanentes
if (Test-Path "auto_hyperontology\khan_academy.py") {
    Remove-Item "auto_hyperontology\khan_academy.py" -Force
}

# Añadir todos los cambios
git add -A

# Commit si hay cambios
git diff-index --quiet HEAD -- 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "📦 Creando commit: '$Message'" -ForegroundColor Yellow
    git commit -m "$Message"
} else {
    Write-Host "ℹ️ No hay cambios pendientes que commitear." -ForegroundColor Gray
}

# Push a GitHub
Write-Host "📤 Empujando cambios a origin main..." -ForegroundColor Green
git push -u origin main

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "✅ ¡Repositorio actualizado exitosamente en GitHub!" -ForegroundColor Green
Write-Host "   🔗 https://github.com/christianemaldonadomti/auto-hyperontology" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
