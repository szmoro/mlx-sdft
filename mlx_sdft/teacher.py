"""EMA teacher wrapper.

The teacher is an exponential moving average of the student weights.
It is always frozen (no optimizer state, no gradient accumulation).

Usage::

    from mlx_sdft import EMATeacher

    # teacher_model is a pre-loaded, frozen copy of the student
    teacher = EMATeacher(teacher_model, decay=0.999)
    # after each optimizer step:
    teacher.update(student)
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class EMATeacher:
    """Wraps a frozen copy of the student for EMA parameter tracking.

    At init, ``model`` should have identical weights to the student.
    After each optimizer step, call ``update(student)`` to advance the EMA.

    EMA rule: φ_t = α · φ_{t-1} + (1 − α) · θ_t
    where φ are teacher params, θ are student params, α = decay.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.model = model
        self.decay = decay

    def update(self, student: nn.Module) -> None:
        """In-place EMA step: φ ← decay·φ + (1−decay)·θ."""
        s_flat = dict(tree_flatten(student.parameters()))
        t_flat = dict(tree_flatten(self.model.parameters()))
        alpha = self.decay
        merged = [(k, alpha * t_flat[k] + (1.0 - alpha) * s_flat[k]) for k in t_flat]
        self.model.update(tree_unflatten(merged))
        mx.eval(self.model.parameters())

    def save(self, path: str | Path) -> None:
        """Save teacher weights to an .npz file."""
        flat = dict(tree_flatten(self.model.parameters()))
        mx.savez(str(path), **flat)
