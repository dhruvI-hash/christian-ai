# Activate the project virtual environment (PowerShell)
$venvPath = Join-Path $PSScriptRoot ".." ".venv" | Resolve-Path
& "$venvPath\Scripts\Activate.ps1"
Write-Host "Activated: $venvPath"
