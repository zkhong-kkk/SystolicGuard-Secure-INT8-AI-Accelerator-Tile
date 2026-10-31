"""Algorithm-to-RTL adapter for the secure INT8 AI accelerator tile.

PyTorch is used on CUDA when available, otherwise CPU. The same quantized
layer is exported to a file-driven Verilog testbench and checked bit-exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def parse_values(text: str, count: int, name: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",")]
    if len(values) != count:
        raise ValueError(f"{name} requires {count} comma-separated integers")
    return values


def pure_matmul(a: list[int], b: list[int], bias: list[int]) -> list[int]:
    return [
        a[0] * b[0] + a[1] * b[2] + bias[0],
        a[0] * b[1] + a[1] * b[3] + bias[1],
        a[2] * b[0] + a[3] * b[2] + bias[0],
        a[2] * b[1] + a[3] * b[3] + bias[1],
    ]


def requantize(value: int, shift: int, relu: bool) -> int:
    value >>= shift
    if relu and value < 0:
        value = 0
    return max(-128, min(127, value))


def torch_reference(a: list[int], b: list[int], bias: list[int], backend: str):
    try:
        import torch
    except ImportError:
        if backend == "cuda":
            raise RuntimeError("CUDA backend requested but PyTorch is not installed")
        return pure_matmul(a, b, bias), "python-cpu"

    cuda_ok = torch.cuda.is_available()
    if backend == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    device = "cuda" if backend == "cuda" or (backend == "auto" and cuda_ok) else "cpu"
    ta = torch.tensor(a, dtype=torch.float32, device=device).reshape(2, 2)
    tb = torch.tensor(b, dtype=torch.float32, device=device).reshape(2, 2)
    t_bias = torch.tensor(bias, dtype=torch.float32, device=device)
    result = (ta @ tb + t_bias).to("cpu").to(torch.int32).flatten().tolist()
    expected = pure_matmul(a, b, bias)
    if result != expected:
        raise RuntimeError(f"PyTorch mismatch: {result} != {expected}")
    label = f"torch-{device}"
    if device == "cuda":
        label += f" ({torch.cuda.get_device_name(0)})"
    return result, label


def unsigned32(value: int) -> int:
    return value & 0xFFFFFFFF


def export_vectors(words: list[int]) -> Path:
    BUILD.mkdir(exist_ok=True)
    vector_file = BUILD / "adapter_vectors.hex"
    vector_file.write_text("".join(f"{unsigned32(v):08x}\n" for v in words), encoding="ascii")
    return vector_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one layer on PyTorch and RTL")
    parser.add_argument("--a", default="1,2,3,4", help="row-major 2x2 activations")
    parser.add_argument("--b", default="5,6,7,8", help="row-major 2x2 weights")
    parser.add_argument("--bias", default="5,-10", help="two output-channel biases")
    parser.add_argument("--shift", type=int, default=1, choices=range(0, 32))
    parser.add_argument("--no-relu", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    a = parse_values(args.a, 4, "A")
    b = parse_values(args.b, 4, "B")
    bias = parse_values(args.bias, 2, "bias")
    if any(not -128 <= value <= 127 for value in a + b):
        raise ValueError("A and B values must fit signed INT8")

    raw = pure_matmul(a, b, bias)
    relu = not args.no_relu
    full = [max(0, value) if relu else value for value in raw]
    quant = [requantize(value, args.shift, relu) for value in raw]
    control = (args.shift << 1) | int(relu)
    words = a + b + bias + [control] + full + quant
    vector_file = export_vectors(words)
    torch_raw, backend = torch_reference(a, b, bias, args.backend)
    if torch_raw != raw:
        raise RuntimeError(f"backend result mismatch: {torch_raw} != {raw}")

    print(f"Backend: {backend}")
    print(f"A={a} B={b} bias={bias} shift={args.shift} relu={relu}")
    print(f"C32={full}")
    print(f"C8 ={quant}")
    print(f"VECTOR_EXPORT_PASS: {vector_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
