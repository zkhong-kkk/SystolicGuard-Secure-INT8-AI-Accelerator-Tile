"""Software golden model for the INT8 AI accelerator architecture."""

import random


def run_tile(a, b, bias=(0, 0), shift=0, relu=True):
    raw = [
        a[0] * b[0] + a[1] * b[2] + bias[0],
        a[0] * b[1] + a[1] * b[3] + bias[1],
        a[2] * b[0] + a[3] * b[2] + bias[0],
        a[2] * b[1] + a[3] * b[3] + bias[1],
    ]
    full = [max(0, value) if relu else value for value in raw]
    quant = []
    for value in raw:
        value >>= shift
        if relu:
            value = max(0, value)
        quant.append(max(-128, min(127, value)))
    return full, quant


def main():
    full, quant = run_tile([1, 2, 3, 4], [5, 6, 7, 8], [5, -10], 1, True)
    assert full == [24, 12, 48, 40]
    assert quant == [12, 6, 24, 20]
    random.seed(2026)
    for _ in range(1000):
        a = [random.randint(-8, 7) for _ in range(4)]
        b = [random.randint(-8, 7) for _ in range(4)]
        bias = [random.randint(-16, 15) for _ in range(2)]
        full, quant = run_tile(a, b, bias, random.randint(0, 3), bool(random.getrandbits(1)))
        assert all(-(2**31) <= value < 2**31 for value in full)
        assert all(-128 <= value <= 127 for value in quant)
    print("PASS golden model: directed case + 1000 randomized cases")


if __name__ == "__main__":
    main()
