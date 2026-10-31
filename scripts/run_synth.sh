#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p build
yosys -s scripts/synth_yosys.ys
echo "Synthesis report: $ROOT/build/synthesis_report.txt"
