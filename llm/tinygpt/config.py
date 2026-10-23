"""Model size settings. Every number here changes how much the model can learn and
how much GPU memory it needs - the RTX 3050 has 6 GB, so start small."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 256      # number of distinct token IDs (set from the tokenizer)
    block_size: int = 256      # context window: how many previous tokens the model can see
    n_layer: int = 6           # number of Transformer blocks stacked on top of each other
    n_head: int = 6            # attention heads per block (n_embd must divide evenly by this)
    n_embd: int = 384          # size of each token's vector (the model's "width")
    dropout: float = 0.1       # fraction of activations randomly zeroed during training (regularisation)
    bias: bool = True          # use bias terms in Linear/LayerNorm layers
    # True: PyTorch's fused attention kernel (fast). False: the step-by-step version in
    # model.py, which is easier to read. Both compute exactly the same thing.
    fused_attention: bool = True

    def __post_init__(self):
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

    def to_dict(self) -> dict:
        return asdict(self)


# Ready-made sizes. Parameter counts are approximate for a 4k-token vocabulary.
PRESETS: dict[str, dict] = {
    "tiny": dict(block_size=128, n_layer=4, n_head=4, n_embd=128),   # ~1M params: seconds per step on CPU, for tests
    "small": dict(block_size=256, n_layer=6, n_head=6, n_embd=384),  # ~12M params: fits the RTX 3050 comfortably
    "medium": dict(block_size=512, n_layer=8, n_head=8, n_embd=512),  # ~30M params: near the practical limit of 6 GB
}


def preset(name: str, **overrides) -> GPTConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    return GPTConfig(**{**PRESETS[name], **overrides})
