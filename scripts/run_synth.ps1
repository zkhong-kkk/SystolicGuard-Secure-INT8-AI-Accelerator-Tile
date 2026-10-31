$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command yosys -ErrorAction SilentlyContinue)) {
    throw "Yosys was not found in PATH. Run this inside OSS CAD Suite."
}
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "build") | Out-Null
Push-Location $projectRoot
try {
    & yosys -s "scripts/synth_yosys.ys"
    if ($LASTEXITCODE -ne 0) { throw "Synthesis failed." }
} finally { Pop-Location }
Write-Host "Synthesis report: $projectRoot\build\synthesis_report.txt"
