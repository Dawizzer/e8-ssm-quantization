"""
Benchmark: E8 Lattice vs Scalar Quantization on Mamba SSM Hidden States
========================================================================
Core question: Does E8 lattice quantization preserve more information
in SSM hidden states than standard scalar (int) quantization?

E8's geometric advantage is bit-rate dependent — it only emerges at low
bit rates (2-5 bits per dimension) where lattice packing density matters.
This benchmark sweeps across bit rates to find the crossover point.

Run: python benchmark.py
     python benchmark.py --seed 1337
     python benchmark.py --real   (requires mamba-ssm + GPU)
"""

import torch
import numpy as np
import json
from pathlib import Path

from e8_quantizer import E8Quantizer
from capture_states import load_mamba_model, run_inference_with_capture, CapturedStates


# ── Test corpus ──────────────────────────────────────────────────────────────
TEST_TEXTS = [
    "The quantum structure of spacetime may encode information holographically.",
    "Revenue for Q3 exceeded projections by 12% driven by enterprise adoption.",
    "Once upon a time in a land far away there lived a king who had three sons.",
    " ".join(["The system processes inputs sequentially, updating internal state."] * 10),
    " ".join([
        "Neural architectures that maintain fixed-size state representations",
        "offer fundamental computational advantages over attention mechanisms.",
        "The key insight is that information can be compressed into geometric",
        "structures that preserve semantic relationships while reducing memory.",
        "This approach mirrors biological memory consolidation processes."
    ] * 5),
]

# Bit rates to sweep — this is the key experiment
# E8 theory predicts advantage at low bit rates (2-5), not high (8)
BIT_RATES = [2, 3, 4, 5, 6, 8]


def scalar_quantize(tensor: torch.Tensor, bits: float) -> torch.Tensor:
    """Symmetric scalar quantization at arbitrary bit rate."""
    levels  = int(2 ** bits)
    half    = levels // 2 - 1
    max_val = tensor.abs().max()
    scale   = max_val / half if half > 0 else max_val
    q = torch.clamp(torch.round(tensor / scale), -half, half)
    return (q * scale).to(tensor.dtype)


def measure_error(original: torch.Tensor, quantized: torch.Tensor) -> dict:
    diff = original.float() - quantized.float()
    mse  = diff.pow(2).mean().item()
    snr  = (original.float().pow(2).mean() /
            (diff.pow(2).mean() + 1e-10)).log10().item() * 10
    rel  = (diff.abs() / (original.float().abs() + 1e-8)).mean().item()
    return {"mse": mse, "snr_db": snr, "relative_error": rel}


def run_benchmark(use_real_model: bool = False, seed: int = 42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}\n")

    # ── Validate quantizer first ─────────────────────────────────────────────
    print("Validating E8 quantizer...")
    e8 = E8Quantizer()
    v  = e8.validate(n_samples=10000, seed=seed)
    status = "PASS" if v["pass"] else "FAIL"
    print(f"  Cosets — Integer: {v['pct_integer_coset']*100:.1f}%  "
          f"Half-integer: {v['pct_half_int_coset']*100:.1f}%  "
          f"Violations: {v['violation_rate']*100:.3f}%  [{status}]")
    if not v["pass"]:
        print("  Quantizer invalid — aborting.")
        return

    # ── Generate / load states ───────────────────────────────────────────────
    if use_real_model:
        print("\nLoading Mamba-130M...")
        model, tokenizer = load_mamba_model()
        state_batches = []
        for i, text in enumerate(TEST_TEXTS):
            print(f"  Inference {i+1}/{len(TEST_TEXTS)}: {text[:50]}...")
            _, captured = run_inference_with_capture(model, tokenizer, text)
            state_batches.append(captured.all_states())
    else:
        print("Synthetic mode — generating SSM-like hidden states...")
        print("(Run with --real after installing mamba-ssm for real Mamba states)\n")
        state_batches = []
        for text in TEST_TEXTS:
            n_tokens = len(text.split())
            scale    = 0.5 + np.random.rand() * 1.5
            states   = torch.randn(24, n_tokens, 16) * scale
            states  += torch.sin(torch.linspace(0, 3.14, 16)).unsqueeze(0).unsqueeze(0) * 0.3
            state_batches.append(states.reshape(-1))

    # ── Bit rate sweep ───────────────────────────────────────────────────────
    print(f"{'Bits':>5} | {'E8 SNR':>8} | {'Scalar SNR':>10} | {'E8 advantage':>13} | Verdict")
    print("-" * 62)

    sweep_results = []

    for bits in BIT_RATES:
        e8_snrs     = []
        scalar_snrs = []

        for states in state_batches:
            states = states.float()

            e8.calibrate_scale(states, target_bits_per_dim=bits)
            q_e8   = e8.quantize(states)
            err_e8 = measure_error(states, q_e8)

            q_sc   = scalar_quantize(states, bits)
            err_sc = measure_error(states, q_sc)

            e8_snrs.append(err_e8["snr_db"])
            scalar_snrs.append(err_sc["snr_db"])

        mean_e8 = np.mean(e8_snrs)
        mean_sc = np.mean(scalar_snrs)
        delta   = mean_e8 - mean_sc

        if delta > 1.0:
            verdict = "E8 WINS"
        elif delta > 0.1:
            verdict = "E8 marginal"
        elif delta > -0.1:
            verdict = "tied"
        else:
            verdict = "scalar wins"

        print(f"{bits:>5} | {mean_e8:>8.2f} | {mean_sc:>10.2f} | {delta:>+13.2f} | {verdict}")
        sweep_results.append({
            "bits": bits,
            "e8_mean_snr_db": mean_e8,
            "scalar_mean_snr_db": mean_sc,
            "e8_advantage_db": delta,
            "verdict": verdict
        })

    # ── Crossover analysis ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    crossover = [r for r in sweep_results if r["e8_advantage_db"] > 0.1]
    if crossover:
        low = min(r["bits"] for r in crossover)
        print(f"  E8 advantage appears at {low} bits/dim and below.")
        print(f"  Hypothesis supported in low-bit regime.")
    else:
        print(f"  E8 shows no advantage across tested bit rates.")
        print(f"  Hypothesis not supported on synthetic data.")
        print(f"  Next step: test on real Mamba states (--real flag).")

    out = {"seed": seed, "mode": "real" if use_real_model else "synthetic",
           "bit_rate_sweep": sweep_results}
    with open("benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to benchmark_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Use real Mamba-130M model (requires mamba-ssm + GPU)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    run_benchmark(use_real_model=args.real, seed=args.seed)
