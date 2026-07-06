# Launch YTScribe (GUI). For headless use:  .\run.ps1 --cli <url>
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Virtual environment missing - run setup.ps1 first." -ForegroundColor Red
    exit 1
}
& $venvPy -m ytscribe @args
