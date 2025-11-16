<#
Run the camera preview (testing.run_camera) with the project's venv if present.
Place this script in the `SignLanguageDetectionWorkshop-master` folder and double-click
or run from PowerShell. It will try to activate the venv at the repo root (`..\.venv`).
#>
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')
$venvActivate = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'

if (Test-Path $venvActivate) {
    Write-Host "Activating venv at $venvActivate"
    . $venvActivate
} else {
    Write-Host "No .venv found at $venvActivate — will use system python if available"
}

Set-Location $scriptDir
Write-Host "Starting camera preview (testing.run_camera)..."
python -c "import testing; testing.run_camera()"
