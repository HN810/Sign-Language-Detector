@echo off
REM Simple helper to start the Flask backend from cmd.exe
set HOST=127.0.0.1
set PORT=5000
echo Starting server on http://%HOST%:%PORT%
python server.py
