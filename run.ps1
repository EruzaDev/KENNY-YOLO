param(
    [int]$Port = 8001
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '4' }
    if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = '4' }
    & py -3.11 -m app.serve --port $Port
    if ($LASTEXITCODE -ne 0) { throw "App exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
