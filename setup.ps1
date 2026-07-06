# YTScribe one-time setup for a fresh Windows machine.
# Installs Python 3.12 + FFmpeg via winget (signed installers — safe under
# Smart App Control), creates a virtual environment, installs dependencies.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
}

# --- Python 3.12 ---
$pyExe = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $pyExe)) {
    Write-Host "Installing Python 3.12 (winget)..." -ForegroundColor Cyan
    winget install --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
}
if (-not (Test-Path $pyExe)) { throw "Python 3.12 install failed — install it manually from python.org and re-run." }

# --- FFmpeg ---
Refresh-Path
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Installing FFmpeg (winget)..." -ForegroundColor Cyan
    winget install --id Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
}

# --- virtual environment ---
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    & $pyExe -m venv (Join-Path $root ".venv")
}

Write-Host "Installing dependencies (several GB, one time)..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
# CUDA-enabled PyTorch; falls back to CPU automatically at runtime if no NVIDIA GPU
& $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu126
& $venvPy -m pip install -r (Join-Path $root "requirements.txt")

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
& $venvPy -c "import torch; print('CUDA available:', torch.cuda.is_available())"

Write-Host ""
Write-Host "Setup complete. Start the app with:  .\run.ps1" -ForegroundColor Green
