# Simple helper to start the Flask backend from PowerShell
# Usage: Right-click -> Run with PowerShell or run in a terminal

$env:HOST = '127.0.0.1'
$env:PORT = '5000'
Write-Host "Starting server on http://$($env:HOST):$($env:PORT)"
python server.py
