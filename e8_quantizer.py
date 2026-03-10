"""
E8 Lattice Quantizer
====================
Implements nearest-neighbour E8 lattice quantization for SSM hidden states.

E8 lattice definition:
    { x ∈ R^8 : all coords integers OR all coords half-integers,
                AND sum(coords) is even }

240 minimal vectors, densest sphere packing in R^8 (Viazovska 2016),
optimal mean-squared quantization error for 8D Gaussian sources.

Algorithm: Conway-Sloane nearest-neighbour decoder
    1. Find closest point in integer coset (D8): round to integers, fix parity
    2. Find closest point in half-integer coset: round to half-integers, fix parity
    3. Return whichever coset point is closer to x

Parity fix: if rounded sum is odd, flip the coordinate with the largest
rounding error by ±1 (toward x) to restore even sum. Single operation,
no cascading corrections.

Usage:
    q = E8Quantizer()
    q.calibrate_scale(states)
    quantized = q.quantize(states)
    metrics = q.reconstruction_error(states, quantized)
"""

import torch
import numpy as np


class E8Quantizer:
    def __init__(self, scale: float = 1.0):
        """
        scale: maps real values into E8 lattice space.
               E8 Voronoi cell radius = 1/sqrt(2) ≈ 0.707.
               Calibrate so typical state values land in [-2, 2] lattice units.
               Use calibrate_scale() to set automatically.
        """
        self.scale = scale

    def _nearest_integer_coset(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nearest point in D8 (integer vectors with even coordinate sum).
        x: (..., 8)
        """
        # Round to nearest integer
        y = torch.round(x)

        # Check parity — sum must be even
        s = y.sum(dim=-1) % 2  # (...,) — 0 if even, 1 if odd

        # Where parity is wrong, flip the coord with largest rounding error
        # Flip direction: toward x (minimises additional error)
        err = (x - y).abs()                              # (..., 8)
        worst = err.argmax(dim=-1, keepdim=True)         # (..., 1)

        # Direction: +1 if x > y at that coord, -1 otherwise
        direction = torch.sign(x.gather(-1, worst))
        direction = torch.where(direction == 0,
                                torch.ones_like(direction), direction)

        # Apply fix only where parity is odd
        fix = direction * s.unsqueeze(-1).float()        # (..., 1)
        y = y.scatter_add(-1, worst, fix)

        return y

    def _nearest_half_integer_coset(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nearest point in D8 + (1/2,...,1/2) (half-integer vectors, even sum).
        x: (..., 8)
        """
        # Round to nearest half-integer: floor(x) + 0.5
        y = torch.floor(x) + 0.5

        # Parity check on integer parts (floor values)
        # sum(floor(x_i)) must be even for the point to be in E8
        s = torch.floor(x).sum(dim=-1) % 2              # (...,)

        # Rounding error: |x - (floor(x) + 0.5)|
        err = (x - y).abs()                              # (..., 8)
        worst = err.argmax(dim=-1, keepdim=True)         # (..., 1)

        # Flip worst coord by ±1 (moves it to adjacent half-integer)
        # Direction: toward x
        direction = torch.sign(x.gather(-1, worst))
        direction = torch.where(direction == 0,
                                torch.ones_like(direction), direction)

        fix = direction * s.unsqueeze(-1).float()
        y = y.scatter_add(-1, worst, fix)

        return y

    def _nearest_e8_point(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nearest E8 lattice point to each 8D vector in x.
        x: (..., 8)
        """
        assert x.shape[-1] == 8, f"E8 requires 8D vectors, got {x.shape[-1]}"

        yi = self._nearest_integer_coset(x)
        yh = self._nearest_half_integer_coset(x)

        # Pick whichever coset point is closer
        dist_i = (x - yi).pow(2).sum(dim=-1, keepdim=True)
        dist_h = (x - yh).pow(2).sum(dim=-1, keepdim=True)
        use_int = (dist_i <= dist_h).float()

        return use_int * yi + (1.0 - use_int) * yh

    def quantize(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Quantize hidden_state using E8 lattice.
        Accepts any shape — flattens, tiles into 8D chunks, quantizes, restores shape.
        """
        original_shape = hidden_state.shape
        dtype = hidden_state.dtype

        # Scale into lattice space
        x = hidden_state.float().reshape(-1) / self.scale

        # Pad to multiple of 8
        n = x.shape[0]
        pad = (8 - n % 8) % 8
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))

        # Reshape to (N, 8), quantize, flatten back
        x_8d = x.reshape(-1, 8)
        q_8d = self._nearest_e8_point(x_8d)
        q = q_8d.reshape(-1)

        if pad > 0:
            q = q[:n]

        return (q * self.scale).to(dtype).reshape(original_shape)

    def reconstruction_error(
        self, original: torch.Tensor, quantized: torch.Tensor
    ) -> dict:
        """MSE, MAE, relative error, and SNR in dB."""
        diff = original.float() - quantized.float()
        mse  = diff.pow(2).mean().item()
        mae  = diff.abs().mean().item()
        rel  = (diff.abs() / (original.float().abs() + 1e-8)).mean().item()
        snr  = (
            original.float().pow(2).mean() /
            (diff.pow(2).mean() + 1e-10)
        ).log10().item() * 10
        return {"mse": mse, "mae": mae, "relative_error": rel, "snr_db": snr}

    def calibrate_scale(self, sample_states: torch.Tensor,
                        target_bits_per_dim: float = 8.0) -> float:
        """
        Set scale to match a target bit rate per dimension.

        E8's geometric advantage (~1 dB over scalar quantization per dimension)
        only appears in the fine-quantization regime — many lattice points
        covering the signal range. Too coarse and all methods perform equally
        badly; E8 offers no advantage there.

        Default target_bits_per_dim=8 matches int8, making the comparison fair.
        Int8 step size = max_val / 127 ≈ 3*std / 127.
        E8 scale is set to match that step size so both quantizers operate
        at equivalent resolution.
        """
        x = sample_states.float()
        max_val = x.abs().max().item()
        std = x.std().item()

        if target_bits_per_dim == 8:
            # Match int8: step = max_val / 127
            self.scale = max(max_val / 127.0, 1e-6)
        else:
            # General: levels = 2^bits, step = 6*std / levels
            levels = 2 ** target_bits_per_dim
            self.scale = max(6.0 * std / levels, 1e-6)

        return self.scale

    def validate(self, n_samples: int = 10000, seed: int = 0) -> dict:
        """
        Self-test: verify the quantizer actually produces E8 lattice points.
        Checks that output points satisfy the E8 membership conditions.
        Returns pass/fail and violation rate.
        """
        torch.manual_seed(seed)
        x = torch.randn(n_samples, 8)
        y = self._nearest_e8_point(x)

        # Check 1: all coords either all-integer or all-half-integer per vector
        frac = (y % 1.0).abs()
        is_integer     = (frac < 1e-4).all(dim=-1)
        is_half_int    = ((frac - 0.5).abs() < 1e-4).all(dim=-1)
        coset_valid    = (is_integer | is_half_int)

        # Check 2: sum is even
        sum_even = (y.sum(dim=-1).round() % 2).abs() < 1e-4

        violations = (~(coset_valid & sum_even)).float().mean().item()

        return {
            "violation_rate": violations,
            "pct_integer_coset": is_integer.float().mean().item(),
            "pct_half_int_coset": is_half_int.float().mean().item(),
            "pass": violations < 0.001
        }
