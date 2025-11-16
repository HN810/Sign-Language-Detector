@echo off
:: One-click helper to run the camera preview (testing.run_camera).
:: Place this file in the SignLanguageDetectionWorkshop-master folder and double-click it.

SET SCRIPT_DIR=%~dp0
SET PARENT=%SCRIPT_DIR%..\
SET VENV_ACT=%PARENT%\.venv\Scripts\activate.bat

IF EXIST "%VENV_ACT%" (
  echo Activating venv at %VENV_ACT%
  call "%VENV_ACT%"
) ELSE (
  echo No .venv activate script found, using system python if available
)

cd /d "%SCRIPT_DIR%"

echo Starting camera preview (testing.run_camera)...
python -c "import testing; testing.run_camera()"

pause