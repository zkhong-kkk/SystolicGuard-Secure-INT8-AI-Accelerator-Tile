#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python software/architecture_adapter.py "$@"
iverilog -g2012 -Wall -o build/adapter.vvp \
  rtl/mac_pe.v rtl/systolic_array_2x2.v rtl/relu.v rtl/requantize_int8.v \
  rtl/ai_accelerator_top.v tb/adapter_tb.v
vvp build/adapter.vvp
