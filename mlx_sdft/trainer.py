"""SDFTTrainer: high-level training API for Self-Distillation Fine-Tuning.

Usage::

    from mlx_sdft import SDFTTrainer

    trainer = SDFTTrainer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        teacher_decay=0.999,
        lr=5e-6,
        grad_accum=4,
        max_new_tokens=256,
    )
    trainer.fit(pairs, steps=500)
    trainer.save("./checkpoints/run1")
    trainer.push_to_hub("szmoro/my-sdft-model")

Subclassing for custom losses::

    class CustomSDFTTrainer(SDFTTrainer):
        def _make_loss_fn(self):
            def loss_fn(model, batch):
                ...
            return loss_fn
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Callable, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

from .data import infinite_batches
from .losses import sdft_loss
from .rollout import sample_rollout
from .teacher import EMATeacher


def _grad_norm(grads) -> float:
    flat = [v for _, v in tree_flatten(grads) if isinstance(v, mx.array)]
    if not flat:
        return 0.0
    total = sum(float(mx.sum(g * g).item()) for g in flat)
    return math.sqrt(total)


def _ver(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def _peak_mem_gb() -> float:
    try:
        return mx.get_active_memory() / 1e9
    except Exception:
        return 0.0


class SDFTTrainer:
    """Trains a language model with the SDFT reverse-KL objective.

    Parameters
    ----------
    student:
        The model to train (receives gradient).
    tokenizer:
        HuggingFace tokenizer compatible with the model.
    teacher:
        ``EMATeacher`` wrapping a frozen copy of the student's initial weights.
        If ``None``, raises — use ``from_pretrained`` which creates it automatically.
    lr, weight_decay, beta1, beta2:
        AdamW hyperparameters.
    grad_accum:
        Number of micro-steps per optimizer update.
    grad_clip:
        Maximum gradient norm (pre-clip). Set to 0 to disable.
    ema_decay:
        EMA decay for the teacher (α in φ ← αφ + (1−α)θ).
    max_new_tokens:
        Hard cap on student rollout length.
    sampler_temp, sampler_top_p:
        Rollout sampling parameters.
    model_id:
        HuggingFace model ID; stored in checkpoints for reproducibility.
    """

    def __init__(
        self,
        student: nn.Module,
        tokenizer,
        *,
        teacher: EMATeacher | None = None,
        lr: float = 5e-6,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_accum: int = 4,
        grad_clip: float = 1.0,
        ema_decay: float = 0.999,
        max_new_tokens: int = 256,
        sampler_temp: float = 1.0,
        sampler_top_p: float = 0.9,
        model_id: str | None = None,
    ) -> None:
        if teacher is None:
            raise ValueError(
                "teacher must be provided. Use SDFTTrainer.from_pretrained() "
                "to load student + teacher from a model ID automatically."
            )
        self.student = student
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_accum = grad_accum
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.max_new_tokens = max_new_tokens
        self.sampler_temp = sampler_temp
        self.sampler_top_p = sampler_top_p
        self.model_id = model_id

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        teacher_decay: float = 0.999,
        lr: float = 5e-6,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_accum: int = 4,
        grad_clip: float = 1.0,
        max_new_tokens: int = 256,
        sampler_temp: float = 1.0,
        sampler_top_p: float = 0.9,
    ) -> "SDFTTrainer":
        """Load student + teacher from HuggingFace and return a ready-to-train trainer.

        Both student and teacher are initialised from ``model_id``.
        The teacher is immediately frozen (no optimizer state, no gradient).
        """
        from mlx_lm import load

        print(f"[SDFTTrainer] Loading student from {model_id} …")
        student, tokenizer = load(model_id)
        student.train()

        print(f"[SDFTTrainer] Loading teacher (EMA copy) from {model_id} …")
        teacher_model, _ = load(model_id)
        teacher_model.freeze()
        mx.eval(teacher_model.parameters())
        teacher = EMATeacher(teacher_model, decay=teacher_decay)

        return cls(
            student,
            tokenizer,
            teacher=teacher,
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            grad_accum=grad_accum,
            grad_clip=grad_clip,
            ema_decay=teacher_decay,
            max_new_tokens=max_new_tokens,
            sampler_temp=sampler_temp,
            sampler_top_p=sampler_top_p,
            model_id=model_id,
        )

    def _make_loss_fn(self) -> Callable:
        """Return ``fn(model, batch) -> (scalar_loss, aux)``.

        Override this in subclasses to plug in custom objectives.
        The function signature must match what ``nn.value_and_grad`` expects:
        first argument is the model whose parameters are differentiated.
        """
        teacher_model = self.teacher.model
        tokenizer = self.tokenizer
        max_new_tokens = self.max_new_tokens
        temp = self.sampler_temp
        top_p = self.sampler_top_p

        def loss_fn(model, batch):
            y_ids = sample_rollout(
                model, tokenizer, batch["x_ids"], max_new_tokens, temp, top_p
            )
            return sdft_loss(model, teacher_model, batch["x_ids"], batch["c_ids"], y_ids), y_ids

        return loss_fn

    def fit(
        self,
        pairs: list[dict] | Iterable[dict],
        steps: int,
        *,
        log_every: int = 10,
        save_every: int = 250,
        output_dir: str | Path | None = None,
    ) -> list[dict]:
        """Run SDFT training.

        pairs:
            Either a ``list[{"x_ids": ..., "c_ids": ...}]`` of pre-tokenised pairs
            (output of ``tokenize_pairs`` / ``load_ultrachat``), or any iterable of
            the same format. If a list is passed, it is wrapped in ``infinite_batches``
            for you; iterables are consumed directly.
        steps:
            Number of optimizer updates.
        log_every:
            Append a metrics record to ``metrics.jsonl`` every N optimizer steps.
        save_every:
            Save a checkpoint every N optimizer steps (only when loss improves).
        output_dir:
            Where to write ``metrics.jsonl``, ``run.json``, and checkpoints.
            Defaults to ``./outputs``.

        Returns the list of metrics dicts written during training.
        """
        output_dir = Path(output_dir or "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)

        # Build data iterator
        if isinstance(pairs, list):
            batch_iter: Iterable[dict] = infinite_batches(pairs)
        else:
            batch_iter = iter(pairs)

        optimizer = optim.AdamW(
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )

        loss_fn = self._make_loss_fn()
        vg_fn = nn.value_and_grad(self.student, loss_fn)

        metrics_log: list[dict] = []
        metrics_path = output_dir / "metrics.jsonl"
        accum_grads = None
        accum_count = 0
        opt_step = 0
        micro_step = 0
        loss_f = float("nan")
        gnorm = 0.0
        best_loss = float("inf")
        t0 = time.time()

        pbar = tqdm(total=steps, desc="sdft", unit="step", dynamic_ncols=True)

        while opt_step < steps:
            batch = next(batch_iter)
            (loss_val, _), grads = vg_fn(self.student, batch)
            mx.eval(loss_val, grads)

            # Accumulate gradients
            if accum_grads is None:
                accum_grads = grads
            else:
                flat_acc = dict(tree_flatten(accum_grads))
                flat_new = dict(tree_flatten(grads))
                accum_grads = tree_unflatten(
                    [(k, flat_acc[k] + flat_new[k]) for k in flat_acc]
                )
            accum_count += 1
            micro_step += 1

            if accum_count < self.grad_accum:
                continue

            # Scale + optional clip
            scale = 1.0 / self.grad_accum
            scaled = tree_unflatten(
                [(k, v * scale) for k, v in tree_flatten(accum_grads)]
            )
            gnorm = _grad_norm(scaled)
            if self.grad_clip > 0 and gnorm > self.grad_clip:
                clip_scale = self.grad_clip / (gnorm + 1e-6)
                scaled = tree_unflatten(
                    [(k, v * clip_scale) for k, v in tree_flatten(scaled)]
                )

            optimizer.update(self.student, scaled)
            mx.eval(self.student.parameters(), optimizer.state)

            self.teacher.update(self.student)

            accum_grads = None
            accum_count = 0
            opt_step += 1
            loss_f = float(loss_val.item())

            mem_gb = _peak_mem_gb()
            elapsed = time.time() - t0
            tok_per_sec = (micro_step * 256) / max(elapsed, 1e-3)  # rough estimate
            pbar.set_postfix(
                loss=f"{loss_f:.4f}",
                gnorm=f"{gnorm:.3f}",
                mem=f"{mem_gb:.1f}G",
                tps=f"{tok_per_sec:.0f}",
                refresh=False,
            )
            pbar.update(1)

            if opt_step % log_every == 0 or opt_step == 1:
                record = {
                    "step": opt_step,
                    "loss": round(loss_f, 6),
                    "grad_norm": round(gnorm, 6),
                    "tokens_per_sec": round(tok_per_sec, 1),
                    "mem_gb": round(mem_gb, 2),
                    "elapsed_s": round(elapsed, 1),
                }
                metrics_log.append(record)
                with open(metrics_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")

            if opt_step % save_every == 0 and loss_f < best_loss:
                best_loss = loss_f
                self._save_checkpoint(opt_step, output_dir)
                tqdm.write(f"  [ckpt] saved step {opt_step}  loss={best_loss:.4f}")

        pbar.close()

        # Final checkpoint
        self._save_checkpoint(opt_step, output_dir)

        # Reproducibility receipt
        run_record = {
            "model_id": self.model_id,
            "steps": opt_step,
            "config": self._config_dict(),
            "wall_clock_s": round(time.time() - t0, 1),
            "peak_mem_gb": round(_peak_mem_gb(), 2),
            "versions": {
                "mlx": _ver("mlx"),
                "mlx_lm": _ver("mlx-lm"),
                "mlx_sdft": _ver("mlx-sdft"),
            },
        }
        with open(output_dir / "run.json", "w") as fh:
            json.dump(run_record, fh, indent=2)

        print(f"[SDFTTrainer] Done. Outputs in {output_dir}/")
        return metrics_log

    def save(self, path: str | Path) -> None:
        """Save student weights + trainer config to ``path/``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        flat = dict(tree_flatten(self.student.parameters()))
        mx.savez(str(path / "weights.npz"), **flat)
        (path / "config.json").write_text(json.dumps(self._config_dict(), indent=2))

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool = False,
        metrics: dict | None = None,
    ) -> str:
        """Push student weights + config to HuggingFace Hub.

        Returns the repo URL.
        """
        from .hub import push_to_hub, generate_model_card

        flat = dict(tree_flatten(self.student.parameters()))
        config = self._config_dict()
        card = generate_model_card(
            self.model_id or "unknown", repo_id, config, metrics=metrics
        )
        url = push_to_hub(flat, config, repo_id, token=token, private=private, model_card=card)
        print(f"[SDFTTrainer] Pushed to {url}")
        return url

    def _save_checkpoint(self, step: int, output_dir: Path) -> None:
        ckpt_dir = output_dir / "checkpoints" / f"step_{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        flat = dict(tree_flatten(self.student.parameters()))
        mx.savez(str(ckpt_dir / "weights.npz"), **{k: v for k, v in flat.items()})

    def _config_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "grad_accum": self.grad_accum,
            "grad_clip": self.grad_clip,
            "ema_decay": self.ema_decay,
            "max_new_tokens": self.max_new_tokens,
            "sampler_temp": self.sampler_temp,
            "sampler_top_p": self.sampler_top_p,
        }
