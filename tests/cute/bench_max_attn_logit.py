"""Benchmark return_max_attn_logit overhead."""
import sys
import os
# Ensure local source is used, not installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import time
from flash_attn.cute.interface import flash_attn_func


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    return elapsed


def main():
    device = "cuda"
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name()}")

    configs = [
        # (bs, sq, sk, nh, nh_kv, d, causal)
        # DeepSeek-style: d=192, nh=80, nhkv=16
        (2, 4096, 4096, 80, 16, 192, False),
        (2, 4096, 4096, 80, 16, 192, True),
    ]

    print(f"\n{'config':<55} {'baseline (ms)':>14} {'w/ max_logit (ms)':>18} {'overhead':>10}")
    print("-" * 100)

    for bs, sq, sk, nh, nh_kv, d, causal in configs:
        label = f"bs={bs} sq={sq} sk={sk} nh={nh} nhkv={nh_kv} d={d} causal={causal}"

        q = torch.randn(bs, sq, nh, d, device=device, dtype=dtype)
        k = torch.randn(bs, sk, nh_kv, d, device=device, dtype=dtype)
        v = torch.randn(bs, sk, nh_kv, d, device=device, dtype=dtype)

        t_base = bench(lambda: flash_attn_func(q, k, v, causal=causal))
        t_max = bench(lambda: flash_attn_func(q, k, v, causal=causal, return_max_attn_logit=True))

        overhead_pct = (t_max - t_base) / t_base * 100
        print(f"{label:<55} {t_base:>13.3f} {t_max:>17.3f} {overhead_pct:>+9.1f}%", flush=True)


if __name__ == "__main__":
    main()
