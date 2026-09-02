# Updates the moving parts that YouTube breaks over time.
# Run this whenever downloads start failing with HTTP 403 / "unable to
# download video data", or every month or two as routine maintenance.
#   powershell -ExecutionPolicy Bypass -File update-deps.ps1

$ErrorActionPreference = "Stop"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Virtual environment missing - run setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Updating yt-dlp..." -ForegroundColor Cyan
& $venvPy -m pip install -U yt-dlp
& $venvPy -c "import yt_dlp; print('yt-dlp is now', yt_dlp.version.__version__)"

# yt-dlp needs an external JS runtime for YouTube; Deno is its default.
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not (Get-Command deno -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Deno (JavaScript runtime required by yt-dlp)..." -ForegroundColor Cyan
    winget install --id DenoLand.Deno --silent --accept-package-agreements --accept-source-agreements
} else {
    Write-Host "Deno present: $((deno --version) -split "`n" | Select-Object -First 1)"
}

Write-Host ""
Write-Host "Done. Restart YTScribe, then use 'Retry Failed' to requeue the failures." -ForegroundColor Green
