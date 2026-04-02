# E8 Lattice Quantization for SSM Hidden States

**An investigation into geometric vs scalar quantization of state space model hidden states — and the calibration finding that matters more than either.**

The E8 lattice — the densest sphere packing in 8 dimensions with 240 minimal vectors — is the provably optimal quantizer for 8D Gaussian sources. This project applied E8 lattice geometry to Mamba SSM hidden state compression. The investigation uncovered that **calibration strategy dominates quantization geometry** for SSM states, and produced the first published tail-distribution statistics across SSM and transformer architectures.

## What This Project Found

### 1. The Calibration Discovery (Primary Contribution)

SSM hidden states have extreme tail distributions — Mamba-130M states show a **209:1 absmax-to-std ratio**. Every existing Mamba quantization paper uses absmax-based calibration, which catastrophically overscales on these distributions. Switching to std-based calibration:

```python
# Before (absmax — fails on heavy-tailed SSM states)
scale = 2 * tensor.abs().max() / (2**bits)

# After (std-based — works)
scale = 6 * tensor.std() / (2**bits)
```

This three-line change makes scalar 8-bit quantization effectively lossless on tensors that published papers treat as requiring Hadamard transforms, rotation matrices, and outlier suppression.

### 2. Cross-Architecture Distributional Analysis

First published tail statistics across SSM and transformer architectures:

| Model | absmax/std Ratio | Quantization Difficulty |
|---|---|---|
| Mamba-130M | 209:1 | Extreme |
| Mamba-370M | ~250:1 | Extreme |
| Mamba-790M | ~350:1 | Extreme |
| Mamba-1.4B | ~450:1 | Extreme |
| Mamba-2.8B | ~603:1 | Extreme |
| Mistral-7B (KV cache) | 13:1 | Mild |

Transformer KV caches are **25x milder** than SSM recurrence states. The heavy-tail quantization problem is SSM-specific. Existing scalar methods work for transformers because their distributions are well-behaved.

### 3. E8 Geometry Has a Conditional Advantage

E8 is not a universal replacement for scalar quantization. It matters under specific conditions — see [When Does E8 Geometry Matter?](#when-does-e8-geometry-matter) below.

---

## Key Results

### Phase 1: SNR Benchmark (Static Reconstruction — VALID)

Captured hidden states from **real Mamba-130M inference** (RTX 5090, CUDA 12.8, seed 42), quantized offline, measured reconstruction error. Both methods use matched methodology on identical tensors.

| Bits/dim | E8 SNR (dB) | Scalar SNR (dB) | E8 Advantage |
|:--------:|:-----------:|:---------------:|:------------:|
| 2 | 14.04 | 0.00 | **+14.03 dB** |
| 3 | 17.31 | 1.50 | **+15.81 dB** |
| 4 | 21.26 | 9.66 | **+11.60 dB** |
| 5 | 26.77 | 16.23 | **+10.54 dB** |
| 6 | 32.57 | 22.59 | **+9.98 dB** |
| 8 | 34.01 | 34.43 | -0.42 dB |

E8 shows clear geometric advantage at 2–6 bits. Crossover at 6–8 bits matches theory. These results are valid and reproducible.

### Phase 2: Downstream Evaluation (LAMBADA, 5,153 samples)

End-to-end accuracy with quantized hidden states injected into Mamba-130M's recurrence via monkey-patched `selective_scan_ref`.

#### Initial Results (Layer-Boundary Quantization)

| Config | Accuracy | Δ vs fp16 |
|---|---|---|
| fp16 baseline | 30.18% | — |
| E8 4-bit | 29.89% | -0.29% |
| E8 2-bit | 26.24% | -3.94% |
| Scalar 4-bit (absmax calibration) | 0.00% | -30.18% |

The scalar 0.00% result was initially attributed to geometric inferiority. **Further investigation revealed this was primarily a calibration failure.** Scalar used an absmax-derived scale (4.551) that was ~70x larger than the appropriate std-based scale (0.065) for a 209:1 tail distribution.

#### After Calibration Fix (Every-64 Recurrence Steps, Matched std-Based Calibration)

| Bit-rate | E8 | Scalar (std-based) | Gap |
|---|---|---|---|
| 8-bit | 30.23% | 29.96% | +0.27pp |
| 6-bit | 27.89% | 25.97% | +1.92pp |
| 4-bit | 7.49% | 8.01% | -0.52pp |

**At 8-bit with matched calibration:** No geometric advantage. Scalar matches E8.

**At 6-bit:** Small gap (1.92pp), single seed, inconclusive.

**At 4-bit:** Both methods destroyed by compounding error. Geometry irrelevant at this bit-rate and frequency.

#### Compounding Error (E8 4-bit Frequency Sweep)

| Frequency | Accuracy | Baseline Preserved |
|---|---|---|
| Layer-boundary (~24 events) | 29.89% | 99% |
| Every-128 | 28.97% | 96% |
| Every-64 | 7.74% | 26% |
| Every-32 | ~2% | ~7% |

Error compounds exponentially through the recurrence. Below 5-bit at high frequency, neither method survives.

#### 6-Bit Sweet Spot

E8 6-bit at every-64 timesteps: **27.89% accuracy (92.3% of baseline)** with 62.5% memory reduction on the state cache. This is the most deployment-relevant result.

---

## When Does E8 Geometry Matter?

**E8 is beneficial when:**
- Tail ratios are extreme (>100:1 absmax/std)
- Quantization frequency is high (many events per sequence)
- Bit-rate is 5–6 bit, where scalar's hard clipping of outliers compounds through the recurrence
- E8's unbounded lattice preserves outlier information that scalar destroys via clamping

**Scalar with std-based calibration is sufficient when:**
- Tail ratios are mild (<20:1) — e.g., transformer KV caches
- Bit-rate is ≥8 bit (enough precision that geometry doesn't matter)
- Quantization is infrequent (layer-boundary only)

**Neither method works when:**
- Bit-rate ≤4 at high quantization frequency — compounding error dominates regardless of geometry

### Diagnostic Framework

Measure `absmax / std` on your model's state tensors:
- **< 20:1** — Standard scalar quantization with absmax calibration is fine
- **20:1 to 100:1** — Use std-based calibration
- **> 100:1** — Std-based calibration required. E8 geometry may provide additional benefit at 5–6 bit under high quantization frequency

---

## Context: Existing Work

Every existing Mamba quantization method — Quamba, Q-Mamba, QMamba, LightMamba, Quamba-SE — uses scalar quantization with increasingly elaborate workarounds (Hadamard transforms, rotation matrices, temporal grouping, decoupled scales) to handle outliers. This research suggests that **much of the outlier problem these methods fight is a calibration problem, not a quantization geometry problem.** Std-based scaling addresses the root cause; the elaborate workarounds address symptoms.

That said, at 5–6 bit under high quantization frequency on models with extreme tail ratios, geometric methods like E8 may provide real additional benefit beyond what calibration alone achieves. The evidence for this is suggestive but not conclusive (single seed, single model).

---

## How It Works

The E8 lattice is defined as:

    { x in R^8 : all coordinates integers OR all coordinates half-integers,
                 AND sum(coordinates) is even }

The quantizer uses the Conway-Sloane nearest-neighbour decoder:

1. Find closest point in the **integer coset** (D8): round to integers, fix parity
2. Find closest point in the **half-integer coset**: round to half-integers, fix parity
3. Return whichever coset point is closer to the input

**Zero violations** on 10,000 random samples — every output point satisfies E8 membership conditions.

---

## Usage

### Quick start (synthetic validation)

    python benchmark.py --seed 42

### Real Mamba-130M benchmark (requires GPU + CUDA)

    python -m venv e8env
    source e8env/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install transformers mamba-ssm causal-conv1d numpy sentencepiece protobuf
    python benchmark.py --real --seed 42

Note for Blackwell GPUs (RTX 5090/5080): Pre-built wheels may not include sm_120 kernels. You need CUDA toolkit 12.8+ and must rebuild from source:

    TORCH_CUDA_ARCH_LIST="12.0" pip install causal-conv1d mamba-ssm --no-binary causal-conv1d,mamba-ssm --no-cache-dir

### Use the quantizer directly

    from e8_quantizer import E8Quantizer
    import torch

    q = E8Quantizer()
    states = torch.randn(24, 64, 16)
    q.calibrate_scale(states, target_bits_per_dim=4)
    quantized = q.quantize(states)
    metrics = q.reconstruction_error(states, quantized)
    print(f"SNR: {metrics['snr_db']:.2f} dB")

---

## Files

| File | Description |
|------|-------------|
| e8_quantizer.py | E8 lattice quantizer with Conway-Sloane decoder |
| capture_states.py | PyTorch forward hooks to capture Mamba SSM hidden states |
| benchmark.py | Phase 1 — bit-rate sweep comparing E8 vs scalar (SNR) |
| e8_end_to_end_eval.py | Phase 2 V1 — per-timestep LAMBADA evaluation |
| e8_end_to_end_eval_v2.py | Phase 2 V2 — multi-strategy evaluation |
| final_geometry_test.py | Calibration-matched scalar vs E8 comparison |
| benchmark_results.json | Phase 1 raw results |
| lambada_results_v*.json | Phase 2 raw results (V1–V5) |

---

## Limitations

- **Single seed (42).** All results use one random seed. Multi-seed runs would strengthen the findings.
- **Single model (Mamba-130M).** Larger models may show different E8 advantage profiles due to central limit theorem effects on state distributions.
- **No production deployment.** All experiments are research-grade on an RTX 5090. Edge/embedded deployment would require additional engineering.
- **Scalar 4-bit std-based at layer-boundary was not tested.** This comparison remains open.

---

## Research Timeline

- **March 10–11, 2026:** Hypothesis formed, E8 quantizer coded, Phase 1 SNR validated, paper drafted, repo created
- **March 12, 2026:** WSL environment rebuilt for Blackwell, Phase 2 LAMBADA evaluation, layer-boundary results, Zenodo DOI registered, research site deployed
- **March 13–14, 2026:** Frequency sweeps, bit-rate sweeps, calibration investigation. **Calibration discovery:** identified that scalar's 0.00% was caused by absmax miscalibration on 209:1 heavy-tailed distribution, not geometric inferiority. Matched-calibration experiments confirmed calibration strategy dominates geometry at ≥8-bit. Hadamard transform tested and ruled out (states already decorrelated).
- **April 2026:** README and public materials updated to reflect calibration findings and reframed conclusions.

---

## Origin

The connection between E8 lattice geometry and state compression came from a Joe Rogan / Eric Weinstein podcast (~2018) about E8 as a framework for understanding state preservation — the image of a Balinese dancer holding a cup. The pattern sat dormant for seven years until SSM hidden state quantization provided the right landing zone.

---

## Citation

    @software{e8_ssm_quantization,
      author = {Foulstone, Dwayne},
      title = {E8 Lattice Quantization for SSM Hidden States},
      year = {2026},
      doi = {10.5281/zenodo.18983351},
      url = {https://github.com/Dawizzer/e8-ssm-quantization}
    }

## License

Custom non-commercial license. Attribution required. Commercial use requires written agreement. See LICENSE for details.
