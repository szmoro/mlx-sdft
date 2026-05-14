"""mlx-sdft: Self-Distillation Fine-Tuning for Apple Silicon.

MLX implementation of Shenfeld et al. (2026) https://arxiv.org/abs/2601.19897

Functional API (low-level, composable)::

    from mlx_sdft import sdft_loss, sft_loss, sample_rollout, EMATeacher

High-level API (recommended for most users)::

    from mlx_sdft import SDFTTrainer

    trainer = SDFTTrainer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    trainer.fit(pairs, steps=500)
    trainer.push_to_hub("szmoro/my-sdft-model")
"""

from .data import infinite_batches, load_ultrachat, tokenize_pairs
from .losses import sdft_loss, sft_loss
from .rollout import sample_rollout
from .teacher import EMATeacher
from .trainer import SDFTTrainer

__version__ = "0.1.0"

__all__ = [
    # Functional API
    "sdft_loss",
    "sft_loss",
    "sample_rollout",
    "EMATeacher",
    # High-level API
    "SDFTTrainer",
    # Data utilities
    "tokenize_pairs",
    "load_ultrachat",
    "infinite_batches",
    # Meta
    "__version__",
]
