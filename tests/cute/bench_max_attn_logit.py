"""Benchmark return_max_attn_logit overhead.

Tests both symmetric (d_qk == d_v) and asymmetric (d_qk != d_v) head dims
to measure the impact of disabling rescale_threshold for max_attn_logit.
"""
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
        # (bs, sq, sk, nh, nh_kv, d_qk, d_v, causal, label)
        # motif3: asymmetric head dims (d_qk=192, d_v=128)
        (2, 4096, 4096, 80, 16, 192, 128, False, "motif3 non-causal"),
        (2, 4096, 4096, 80, 16, 192, 128, True,  "motif3 causal"),
        # Symmetric head dims (d_qk=d_v=192) for comparison
        (2, 4096, 4096, 80, 16, 192, 192, False, "symmetric non-causal"),
        (2, 4096, 4096, 80, 16, 192, 192, True,  "symmetric causal"),
        # Shorter sequences
        (2, 1024, 1024, 80, 16, 192, 128, True,  "motif3 causal sq=1024"),
        (2, 8192, 8192, 80, 16, 192, 128, True,  "motif3 causal sq=8192"),
    ]

    print(f"\n{'config':<40} {'baseline (ms)':>14} {'w/ max_logit (ms)':>18} {'overhead':>10}")
    print("-" * 85)

    for bs, sq, sk, nh, nh_kv, d_qk, d_v, causal, label in configs:
        q = torch.randn(bs, sq, nh, d_qk, device=device, dtype=dtype)
        k = torch.randn(bs, sk, nh_kv, d_qk, device=device, dtype=dtype)
        v = torch.randn(bs, sk, nh_kv, d_v, device=device, dtype=dtype)

        t_base = bench(lambda: flash_attn_func(q, k, v, causal=causal))
        t_max = bench(lambda: flash_attn_func(q, k, v, causal=causal, return_lse=True, return_max_attn_logit=True))

        overhead_pct = (t_max - t_base) / t_base * 100
        print(f"{label:<40} {t_base:>13.3f} {t_max:>17.3f} {overhead_pct:>+9.1f}%", flush=True)


if __name__ == "__main__":
    main()
