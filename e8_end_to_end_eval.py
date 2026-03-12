"""
E8 End-to-End Evaluation: LAMBADA accuracy with quantized hidden states
========================================================================
Patches Mamba-130M's selective_scan to inject E8 or scalar quantize/dequantize
at every recurrence timestep, then evaluates on LAMBADA.

Usage:
    cd ~/e8-ssm-quantization
    source e8env/bin/activate
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
    python e8_end_to_end_eval.py

Outputs a results table comparing fp16 baseline, E8 4-bit, scalar 4-bit,
E8 2-bit, and scalar 2-bit on LAMBADA last-word prediction accuracy.
"""

import sys
import os
import json
import time
import copy
from datetime import datetime

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# ── Add project dir to path for e8_quantizer ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e8_quantizer import E8Quantizer


# ═══════════════════════════════════════════════════════════════════════
# Scalar quantizer (symmetric uniform) for comparison
# ═══════════════════════════════════════════════════════════════════════

class ScalarQuantizer:
    """Symmetric uniform scalar quantization — the standard baseline."""

    def __init__(self, bits: int = 4):
        self.bits = bits
        self.scale = 1.0

    def calibrate_scale(self, sample_states: torch.Tensor) -> float:
        x = sample_states.float()
        max_val = x.abs().max().item()
        levels = 2 ** self.bits
        self.scale = max(2.0 * max_val / levels, 1e-8)
        return self.scale

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        half_levels = 2 ** (self.bits - 1) - 1
        q = torch.clamp(torch.round(xf / self.scale), -half_levels, half_levels)
        return (q * self.scale).to(dtype)


# ═══════════════════════════════════════════════════════════════════════
# Modified selective_scan_ref with quantization hook
# ═══════════════════════════════════════════════════════════════════════

# Global state — set before running eval
_QUANTIZER = None          # None = fp16 baseline, E8Quantizer or ScalarQuantizer
_CALIBRATION_MODE = False  # If True, collect states instead of quantizing
_COLLECTED_STATES = []


def selective_scan_ref_quantized(
    u, delta, A, B, C, D=None, z=None, delta_bias=None,
    delta_softplus=False, return_last_state=False
):
    """
    Exact copy of mamba_ssm selective_scan_ref with E8/scalar quantization
    injected after each recurrence step.
    """
    global _QUANTIZER, _CALIBRATION_MODE, _COLLECTED_STATES

    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(
                rearrange(B.float(), "... (L two) -> ... L two", two=2)
            )
        if is_variable_C:
            C = torch.view_as_complex(
                rearrange(C.float(), "... (L two) -> ... L two", two=2)
            )
    else:
        B = B.float()
        C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    ys = []

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]

        # ═══ QUANTIZATION INJECTION POINT ═══
        if _CALIBRATION_MODE:
            # Collect a sample every 32 timesteps to avoid OOM
            if i % 32 == 0:
                _COLLECTED_STATES.append(x.detach().cpu().reshape(-1)[:4096])
        elif _QUANTIZER is not None:
            x = _QUANTIZER.quantize(x)
        # ═══════════════════════════════════════

        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# ═══════════════════════════════════════════════════════════════════════
# Monkey-patch: force all Mamba blocks to use reference path
# ═══════════════════════════════════════════════════════════════════════

def patch_mamba_to_use_ref():
    """
    Replace selective_scan_fn with our quantized reference implementation
    everywhere it's used in mamba_ssm.
    """
    import mamba_ssm.ops.selective_scan_interface as ssi
    import mamba_ssm.modules.mamba_simple as ms

    # Patch at the module level
    ssi.selective_scan_fn = selective_scan_ref_quantized
    ssi.selective_scan_cuda_fwd = None  # Force fallback

    # Also patch in mamba_simple if it imported selective_scan_fn directly
    if hasattr(ms, 'selective_scan_fn'):
        ms.selective_scan_fn = selective_scan_ref_quantized

    # Patch any references in the Mamba class itself
    if hasattr(ms, 'Mamba'):
        # Some versions store it as a class attribute
        pass

    print("[PATCH] selective_scan_fn → quantized reference implementation")


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    """Load Mamba-130M and tokenizer."""
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    print("[MODEL] Loading Mamba-130M...")
    model = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", device="cuda", dtype=torch.float16
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"[MODEL] Loaded. Device: cuda, dtype: float16")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Calibration: capture hidden states and set quantizer scales
# ═══════════════════════════════════════════════════════════════════════

def calibrate_quantizers(model, tokenizer, n_samples=20):
    """
    Run a few samples through the model in calibration mode,
    collect hidden states from the recurrence, return calibrated
    E8 and scalar quantizers at various bit rates.
    """
    global _CALIBRATION_MODE, _COLLECTED_STATES, _QUANTIZER

    print("[CALIBRATE] Collecting hidden states from recurrence...")
    _CALIBRATION_MODE = True
    _QUANTIZER = None
    _COLLECTED_STATES = []

    # Use diverse calibration prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog and runs through the forest",
        "In the year 2025, artificial intelligence had transformed every aspect of",
        "The stock market experienced unprecedented volatility as investors reacted to",
        "She walked through the ancient library, her fingers tracing the spines of",
        "The chemical formula for water is H2O, which consists of two hydrogen atoms",
    ]

    with torch.no_grad():
        for p in prompts[:n_samples]:
            ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
            _ = model(ids)

    _CALIBRATION_MODE = False

    if not _COLLECTED_STATES:
        raise RuntimeError("No states collected — patch may not be active")

    all_states = torch.cat(_COLLECTED_STATES, dim=0)
    print(f"[CALIBRATE] Collected {all_states.shape[0]} state samples")
    print(f"[CALIBRATE] State stats: mean={all_states.mean():.4f}, "
          f"std={all_states.std():.4f}, max={all_states.abs().max():.4f}")

    # Build quantizers at each bit rate
    quantizers = {}
    for bits in [2, 4]:
        e8 = E8Quantizer()
        e8.calibrate_scale(all_states, target_bits_per_dim=bits)
        quantizers[f"e8_{bits}bit"] = e8
        print(f"[CALIBRATE] E8 {bits}-bit scale: {e8.scale:.6f}")

        sc = ScalarQuantizer(bits=bits)
        sc.calibrate_scale(all_states)
        quantizers[f"scalar_{bits}bit"] = sc
        print(f"[CALIBRATE] Scalar {bits}-bit scale: {sc.scale:.6f}")

    _COLLECTED_STATES = []  # Free memory
    return quantizers


# ═══════════════════════════════════════════════════════════════════════
# LAMBADA evaluation
# ═══════════════════════════════════════════════════════════════════════

def eval_lambada(model, tokenizer, max_samples=None):
    """
    Evaluate on LAMBADA: predict the last word of each passage.
    Returns accuracy as a float.
    """
    from datasets import load_dataset

    print("[EVAL] Loading LAMBADA dataset...")
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    errors = 0

    print(f"[EVAL] Running on {len(ds)} samples...")
    t0 = time.time()

    with torch.no_grad():
        for i, example in enumerate(ds):
            text = example["text"]

            # Split into context and last word
            words = text.rsplit(" ", 1)
            if len(words) != 2:
                continue
            context, target = words[0], words[1]

            # Strip punctuation from target for fair comparison
            target_clean = target.strip().rstrip(".,!?;:'\")")

            try:
                input_ids = tokenizer(
                    context, return_tensors="pt"
                ).input_ids.cuda()

                output = model(input_ids)
                next_token = output.logits[0, -1].argmax()
                predicted = tokenizer.decode(next_token).strip()

                if predicted == target_clean:
                    correct += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [WARN] Sample {i} error: {e}")

            total += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                acc = correct / total * 100
                rate = total / elapsed
                print(f"  [{i+1}/{len(ds)}] acc={acc:.2f}% "
                      f"({rate:.1f} samples/s)")

    elapsed = time.time() - t0
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"[EVAL] Done: {correct}/{total} = {accuracy:.2f}% "
          f"({elapsed:.1f}s, {errors} errors)")

    return accuracy


# ═══════════════════════════════════════════════════════════════════════
# Main: run all configurations
# ═══════════════════════════════════════════════════════════════════════

def main():
    global _QUANTIZER

    torch.manual_seed(42)
    print("=" * 70)
    print("E8 LATTICE QUANTIZATION — END-TO-END LAMBADA EVALUATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check GPU
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Patch first, then load model
    patch_mamba_to_use_ref()
    model, tokenizer = load_model()

    # Calibrate
    quantizers = calibrate_quantizers(model, tokenizer)

    # Define evaluation configs
    configs = [
        ("fp16 baseline", None),
        ("E8 4-bit",      quantizers["e8_4bit"]),
        ("Scalar 4-bit",  quantizers["scalar_4bit"]),
        ("E8 2-bit",      quantizers["e8_2bit"]),
        ("Scalar 2-bit",  quantizers["scalar_2bit"]),
    ]

    results = {}

    for name, quantizer in configs:
        print(f"\n{'─' * 70}")
        print(f"CONFIG: {name}")
        print(f"{'─' * 70}")

        _QUANTIZER = quantizer
        acc = eval_lambada(model, tokenizer)
        results[name] = acc

        # Clear CUDA cache between runs
        torch.cuda.empty_cache()

    # ── Results table ──
    print(f"\n{'═' * 70}")
    print("RESULTS: LAMBADA Last-Word Prediction Accuracy")
    print(f"{'═' * 70}")
    print(f"{'Config':<20} {'Accuracy':>10} {'Δ vs fp16':>12}")
    print(f"{'─' * 42}")

    baseline = results.get("fp16 baseline", 0)
    for name, acc in results.items():
        delta = acc - baseline
        delta_str = f"{delta:+.2f}%" if name != "fp16 baseline" else "—"
        print(f"{name:<20} {acc:>9.2f}% {delta_str:>12}")

    print(f"{'─' * 42}")

    # ── Save results ──
    output = {
        "date": datetime.now().isoformat(),
        "model": "state-spaces/mamba-130m",
        "task": "lambada_openai",
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
        "quantizer_scales": {
            k: getattr(v, 'scale', None)
            for k, v in quantizers.items()
        }
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lambada_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # ── Verdict ──
    print(f"\n{'═' * 70}")
    e8_4 = results.get("E8 4-bit", 0)
    sc_4 = results.get("Scalar 4-bit", 0)
    e8_2 = results.get("E8 2-bit", 0)
    sc_2 = results.get("Scalar 2-bit", 0)

    print("VERDICT:")
    if abs(e8_4 - baseline) <= 2.0 and (baseline - sc_4) >= 5.0:
        print("  ★ STRONG RESULT: E8 4-bit preserves accuracy, scalar collapses")
    elif abs(e8_4 - baseline) <= 2.0:
        print("  ✓ E8 4-bit preserves accuracy (within 2% of baseline)")
    elif e8_4 > sc_4:
        print("  ~ E8 4-bit outperforms scalar but both degrade")
    else:
        print("  ✗ E8 advantage does not translate to downstream accuracy")

    print(f"  E8 4-bit vs scalar 4-bit gap: {e8_4 - sc_4:+.2f}%")
    print(f"  E8 2-bit vs scalar 2-bit gap: {e8_2 - sc_2:+.2f}%")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
