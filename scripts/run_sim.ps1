$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $projectRoot "build"
if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
    throw "Icarus Verilog (iverilog) was not found in PATH. Install it first."
}
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Push-Location $projectRoot
try {
    & iverilog -g2012 -Wall -o "build/ai_accelerator.vvp" `
        "rtl/mac_pe.v" "rtl/systolic_array_2x2.v" "rtl/relu.v" `
        "rtl/requantize_int8.v" `
        "rtl/ai_accelerator_top.v" "tb/ai_accelerator_tb.v"
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed." }
    & vvp "build/ai_accelerator.vvp"
    if ($LASTEXITCODE -ne 0) { throw "Simulation failed." }
} finally { Pop-Location }
Write-Host "Waveform: $buildDir\ai_accelerator.vcd"
