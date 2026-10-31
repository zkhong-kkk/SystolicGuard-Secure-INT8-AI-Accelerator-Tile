#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p build
iverilog -g2012 -Wall -o build/ai_accelerator.vvp \
  rtl/mac_pe.v rtl/systolic_array_2x2.v rtl/relu.v rtl/requantize_int8.v \
  rtl/ai_accelerator_top.v tb/ai_accelerator_tb.v
vvp build/ai_accelerator.vvp
echo "Waveform: $PROJECT_ROOT/build/ai_accelerator.vcd"
