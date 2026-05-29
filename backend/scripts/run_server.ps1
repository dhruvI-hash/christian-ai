# Run the backend API using the project virtual environment
$backendRoot = Split-Path $PSScriptRoot -Parent
Set-Location $backendRoot
& "$backendRoot\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
