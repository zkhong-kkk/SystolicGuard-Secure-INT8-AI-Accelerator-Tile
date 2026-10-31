$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    python "software/architecture_adapter.py" @args
    if ($LASTEXITCODE -ne 0) { throw "Algorithm/vector export failed." }
    & iverilog -g2012 -Wall -o "build/adapter.vvp" `
        "rtl/mac_pe.v" "rtl/systolic_array_2x2.v" "rtl/relu.v" `
        "rtl/requantize_int8.v" "rtl/ai_accelerator_top.v" "tb/adapter_tb.v"
    if ($LASTEXITCODE -ne 0) { throw "Adapter RTL compilation failed." }
    & vvp "build/adapter.vvp"
    if ($LASTEXITCODE -ne 0) { throw "Adapter RTL simulation failed." }
}
finally { Pop-Location }
