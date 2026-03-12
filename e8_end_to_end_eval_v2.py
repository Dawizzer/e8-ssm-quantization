"""
E8 End-to-End Evaluation V2: Layer-Boundary Quantization
=========================================================
Instead of quantizing inside the recurrence at every timestep (which
compounds error catastrophically), this version quantizes at layer
boundaries — after each Mamba block completes its full recurrence.

This matches real-world deployment: compress the hidden state for
storage/transfer between layers, not mid-computation.

Also tests intermediate strategies:
  - Every-N-steps: quantize inside recurrence but only every N timesteps
  - Layer-boundary: quantize block output only

Usage:
    cd ~/e8-ssm-quantization
    source e8env/bin/activate
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
    python e8_end_to_end_eval_v2.py
"""

import sys
import os
import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e8_quantizer import E8Quantizer


# ═══════════════════════════════════════════════════════════════════════
# Scalar quantizer
# ═══════════════════════════════════════════════════════════════════════

class ScalarQuantizer:
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
# Globals
# ═══════════════════════════════════════════════════════════════════════

_QUANTIZER = None
_QUANT_MODE = "none"       # "none", "every_step", "every_n", "layer_boundary"
_QUANT_EVERY_N = 16        # For "every_n" mode
_CALIBRATION_MODE = False
_COLLECTED_STATES = []


# ═══════════════════════════════════════════════════════════════════════
# Modified selective_scan_ref with configurable quantization
# ═══════════════════════════════════════════════════════════════════════

def selective_scan_ref_quantized(
    u, delta, A, B, C, D=None, z=None, delta_bias=None,
    delta_softplus=False, return_last_state=False
):
    global _QUANTIZER, _QUANT_MODE, _QUANT_EVERY_N
    global _CALIBRATION_MODE, _COLLECTED_STATES

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

        # ═══ QUANTIZATION INJECTION ═══
        if _CALIBRATION_MODE:
            if i % 32 == 0:
                _COLLECTED_STATES.append(x.detach().cpu().reshape(-1)[:4096])
        elif _QUANTIZER is not None:
            if _QUANT_MODE == "every_step":
                x = _QUANTIZER.quantize(x)
            elif _QUANT_MODE == "every_n" and i % _QUANT_EVERY_N == 0:
                x = _QUANTIZER.quantize(x)
            # "layer_boundary" mode: don't quantize inside the loop at all
        # ═══════════════════════════════

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
# Layer-boundary hooks: quantize output of each Mamba mixer
# ═══════════════════════════════════════════════════════════════════════

_HOOKS = []

def install_layer_boundary_hooks(model):
    """Attach forward hooks to each Mamba mixer to quantize its output."""
    global _HOOKS
    remove_hooks()

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if _QUANTIZER is not None and _QUANT_MODE == "layer_boundary":
                if isinstance(output, tuple):
                    quantized = _QUANTIZER.quantize(output[0])
                    return (quantized,) + output[1:]
                else:
                    return _QUANTIZER.quantize(output)
            return output
        return hook_fn

    layer_idx = 0
    for name, module in model.named_modules():
        # Target the Mamba mixer blocks
        if type(module).__name__ == "Mamba":
            h = module.register_forward_hook(make_hook(layer_idx))
            _HOOKS.append(h)
            layer_idx += 1

    print(f"[HOOKS] Installed layer-boundary hooks on {layer_idx} Mamba blocks")


def remove_hooks():
    global _HOOKS
    for h in _HOOKS:
        h.remove()
    _HOOKS = []


# ═══════════════════════════════════════════════════════════════════════
# Patching
# ═══════════════════════════════════════════════════════════════════════

def patch_mamba_to_use_ref():
    import mamba_ssm.ops.selective_scan_interface as ssi
    import mamba_ssm.modules.mamba_simple as ms

    ssi.selective_scan_fn = selective_scan_ref_quantized
    ssi.selective_scan_cuda_fwd = None

    if hasattr(ms, 'selective_scan_fn'):
        ms.selective_scan_fn = selective_scan_ref_quantized

    print("[PATCH] selective_scan_fn → quantized reference implementation")


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_model():
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
# Calibration
# ═══════════════════════════════════════════════════════════════════════

def calibrate_quantizers(model, tokenizer):
    global _CALIBRATION_MODE, _COLLECTED_STATES, _QUANTIZER, _QUANT_MODE

    print("[CALIBRATE] Collecting hidden states...")
    _CALIBRATION_MODE = True
    _QUANTIZER = None
    _QUANT_MODE = "every_step"  # Need this active so the scan loop collects
    _COLLECTED_STATES = []

    prompts = [
        "The quick brown fox jumps over the lazy dog and runs through the forest",
        "In the year 2025, artificial intelligence had transformed every aspect of",
        "The stock market experienced unprecedented volatility as investors reacted to",
        "She walked through the ancient library, her fingers tracing the spines of",
        "The chemical formula for water is H2O, which consists of two hydrogen atoms",
    ]

    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
            _ = model(ids)

    _CALIBRATION_MODE = False

    all_states = torch.cat(_COLLECTED_STATES, dim=0)
    print(f"[CALIBRATE] Collected {all_states.shape[0]} state samples")
    print(f"[CALIBRATE] Stats: mean={all_states.mean():.4f}, "
          f"std={all_states.std():.4f}, max={all_states.abs().max():.4f}")

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

    _COLLECTED_STATES = []
    return quantizers


# ═══════════════════════════════════════════════════════════════════════
# LAMBADA evaluation
# ═══════════════════════════════════════════════════════════════════════

def eval_lambada(model, tokenizer, max_samples=None):
    from datasets import load_dataset

    print("[EVAL] Loading LAMBADA dataset...")
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    errors = 0
    t0 = time.time()

    print(f"[EVAL] Running on {len(ds)} samples...")

    with torch.no_grad():
        for i, example in enumerate(ds):
            text = example["text"]
            words = text.rsplit(" ", 1)
            if len(words) != 2:
                continue
            context, target = words[0], words[1]
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
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    global _QUANTIZER, _QUANT_MODE, _QUANT_EVERY_N

    torch.manual_seed(42)
    print("=" * 70)
    print("E8 LATTICE QUANTIZATION — V2: MULTI-STRATEGY EVALUATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    patch_mamba_to_use_ref()
    model, tokenizer = load_model()
    quantizers = calibrate_quantizers(model, tokenizer)

    # Install hooks for layer-boundary mode
    install_layer_boundary_hooks(model)

    # ── Evaluation configs ──
    # Format: (name, quantizer, mode, every_n)
    configs = [
        # Baseline
        ("fp16 baseline",            None,                       "none",           0),
        # Layer-boundary: quantize block outputs only (24 cycles)
        ("E8 4-bit (layer-bdry)",    quantizers["e8_4bit"],      "layer_boundary", 0),
        ("Scalar 4-bit (layer-bdry)",quantizers["scalar_4bit"],  "layer_boundary", 0),
        ("E8 2-bit (layer-bdry)",    quantizers["e8_2bit"],      "layer_boundary", 0),
        ("Scalar 2-bit (layer-bdry)",quantizers["scalar_2bit"],  "layer_boundary", 0),
        # Every-16-steps inside recurrence
        ("E8 4-bit (every-16)",      quantizers["e8_4bit"],      "every_n",       16),
        ("Scalar 4-bit (every-16)",  quantizers["scalar_4bit"],  "every_n",       16),
    ]

    results = {}

    for name, quantizer, mode, every_n in configs:
        print(f"\n{'─' * 70}")
        print(f"CONFIG: {name}")
        print(f"{'─' * 70}")

        _QUANTIZER = quantizer
        _QUANT_MODE = mode
        _QUANT_EVERY_N = every_n

        acc = eval_lambada(model, tokenizer)
        results[name] = acc
        torch.cuda.empty_cache()

    # ── Remove hooks ──
    remove_hooks()

    # ── Results table ──
    print(f"\n{'═' * 70}")
    print("RESULTS: LAMBADA Last-Word Prediction Accuracy")
    print(f"{'═' * 70}")
    print(f"{'Config':<35} {'Accuracy':>10} {'Δ vs fp16':>12}")
    print(f"{'─' * 57}")

    baseline = results.get("fp16 baseline", 0)
    for name, acc in results.items():
        delta = acc - baseline
        delta_str = f"{delta:+.2f}%" if name != "fp16 baseline" else "—"
        print(f"{name:<35} {acc:>9.2f}% {delta_str:>12}")

    print(f"{'─' * 57}")

    # ── Save ──
    output = {
        "date": datetime.now().isoformat(),
        "model": "state-spaces/mamba-130m",
        "task": "lambada_openai",
        "version": "v2_multi_strategy",
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
        "quantizer_scales": {
            k: getattr(v, 'scale', None)
            for k, v in quantizers.items()
        }
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lambada_results_v2.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # ── Verdict ──
    print(f"\n{'═' * 70}")
    print("VERDICT BY STRATEGY:")

    for strategy in ["layer-bdry", "every-16", "every-step"]:
        e8_key = f"E8 4-bit ({strategy})"
        sc_key = f"Scalar 4-bit ({strategy})"
        if e8_key in results and sc_key in results:
            e8_acc = results[e8_key]
            sc_acc = results[sc_key]
            e8_drop = baseline - e8_acc
            sc_drop = baseline - sc_acc
            print(f"\n  {strategy}:")
            print(f"    E8 4-bit:     {e8_acc:.2f}% (drop: {e8_drop:+.2f}%)")
            print(f"    Scalar 4-bit: {sc_acc:.2f}% (drop: {sc_drop:+.2f}%)")
            print(f"    E8 advantage: {e8_acc - sc_acc:+.2f}%")

    print(f"\n{'═' * 70}")


if __name__ == "__main__":
    main()
