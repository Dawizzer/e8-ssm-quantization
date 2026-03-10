"""
Mamba Hidden State Capture
==========================
Uses PyTorch forward hooks to intercept SSM hidden states during inference.

Targets the selective state space layer (S6) hidden state — the fixed-size
compressed representation that IS the memory fingerprint.

Requires: pip install mamba-ssm transformers torch
Model:     state-spaces/mamba-130m  (small, fast, sufficient for PoC)
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CapturedStates:
    """Container for hidden states captured across layers and timesteps."""
    layer_states: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    token_count: int = 0

    def add(self, layer_name: str, state: torch.Tensor):
        if layer_name not in self.layer_states:
            self.layer_states[layer_name] = []
        self.layer_states[layer_name].append(state.detach().cpu())

    def all_states(self) -> torch.Tensor:
        """Flatten all captured states into a single tensor for analysis."""
        all_s = []
        for states in self.layer_states.values():
            all_s.extend(states)
        return torch.cat([s.reshape(-1) for s in all_s])

    def summary(self) -> str:
        lines = [f"Captured {self.token_count} tokens, {len(self.layer_states)} layers"]
        for name, states in self.layer_states.items():
            shape = states[0].shape if states else "empty"
            lines.append(f"  {name}: {len(states)} captures, shape {shape}")
        return "\n".join(lines)


class MambaStateCapture:
    """
    Attaches hooks to a Mamba model to capture SSM hidden states.

    The SSM state (h) in Mamba is the key object — it's the compressed
    representation of all prior context at each layer. Shape: (batch, d_state, d_inner)
    Fixed size regardless of sequence length — this IS the memory fingerprint.
    """

    def __init__(self, model, target_module_names: Optional[List[str]] = None):
        self.model = model
        self.handles = []
        self.captured = CapturedStates()
        self.target_names = target_module_names  # None = capture all Mamba blocks

    def _make_hook(self, layer_name: str):
        def hook(module, input, output):
            # Mamba block output is typically (hidden_states, residual) or just hidden_states
            # The SSM state is internal — we capture the output hidden state as proxy
            if isinstance(output, tuple):
                state = output[0]
            else:
                state = output
            self.captured.add(layer_name, state)
        return hook

    def attach(self):
        """Attach hooks to all target layers."""
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            # Target Mamba's core SSM layer
            if "Mamba" in module_type or "SSM" in module_type or "Mixer" in module_type:
                if self.target_names is None or any(t in name for t in self.target_names):
                    handle = module.register_forward_hook(self._make_hook(name))
                    self.handles.append(handle)
        print(f"Attached {len(self.handles)} hooks")
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.captured = CapturedStates()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *args):
        self.detach()


def load_mamba_model(model_id: str = "state-spaces/mamba-130m"):
    """
    Load a Mamba model for state capture experiments.
    Falls back gracefully with install instructions if mamba_ssm not found.
    """
    try:
        from transformers import AutoTokenizer, MambaForCausalLM
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = MambaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        model.eval()
        return model, tokenizer
    except ImportError:
        raise ImportError(
            "Install requirements:\n"
            "  pip install mamba-ssm transformers torch\n"
            "  pip install causal-conv1d>=1.1.0\n"
            "  (requires CUDA)"
        )


def run_inference_with_capture(
    model, tokenizer, text: str, device: str = "cuda"
) -> tuple:
    """
    Run inference on text, return (output, captured_states).
    """
    model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    capture = MambaStateCapture(model)

    with capture:
        with torch.no_grad():
            output = model(**inputs)
        captured = capture.captured
        captured.token_count = inputs["input_ids"].shape[1]

    return output, captured
