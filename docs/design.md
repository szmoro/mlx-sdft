# Design Notes

## Why a standalone package, not a PR to mlx-lm?

The mlx-lm maintainers have an explicit, documented pattern of rejecting alignment and distillation training-method PRs:

- PR #794 (DPO) — closed: *"Closing per discussion in #795"*
- PR #795 (ORPO) — closed with redirect to `mlx-lm-lora`
- PR #796 (distributed DPO/ORPO) — closed for same reason

The maintainers want mlx-lm to stay a lean inference library. Training methods with custom loss semantics belong in standalone packages like `mlx-lm-lora` (which hosts DPO, ORPO, GRPO, etc. after the PRs were closed). SDFT is in the same category.

The canonical prior art is `mlx-lm-lora`: flat package layout, functional API + Trainer class, published on PyPI. `mlx-sdft` follows the same genre.

## Why keep the functional API public?

`SDFTTrainer` is the friendly entry point for most users. The functional API (`sdft_loss`, `sft_loss`, `sample_rollout`, `EMATeacher`) is kept public for:

1. **Research extensions** — subclassing `SDFTTrainer` and overriding `_make_loss_fn` requires access to the primitives.
2. **Composability** — users may want to combine SDFT with other objectives or wrap it in a different outer loop.
3. **Composability** — users experimenting with sparse fine-tuning or other extensions can build on the primitives without modifying the Trainer.

## Why `_make_loss_fn` as the subclass hook?

MLX's `nn.value_and_grad(model, fn)` requires `fn(model, ...)` as a free function (not a method). The hook returns a closure that captures trainer state, which is passed to `value_and_grad` once at the start of `fit`. This:

- Avoids recreating the autograd graph every step (efficient).
- Lets subclasses inject different objectives without touching the training loop.
- Keeps the teacher reference inside the closure so it can't be accidentally swapped mid-run.

A `compute_loss(self, batch)` method hook was considered, but it would require wrapping in a free function wrapper every call, which has non-trivial overhead in MLX due to the lazy evaluation model.

## Apple Silicon performance notes

- Two model copies (student + teacher) fit in 32 GB unified memory for 1.5B parameters: ~3 GB weights × 2 + ~6 GB AdamW optimizer state + ~4 GB OS/runtime overhead ≈ 16–20 GB.
- The `3B` model does not fit — AdamW state (m + v) for 3B parameters pushes the total to ~37 GB, which causes swap-induced slowdown (222 s/step vs 14 s/step for 1.5B).
- `stream_generate` does not support autograd; rollouts must be sampled outside the autograd graph and re-fed through teacher-forced forwards. This is why `sample_rollout` is a separate function that runs fully detached.
- `mx.get_active_memory()` replaces the deprecated `mx.metal.get_active_memory()` (changed in mlx ≥ 0.31).

## Teacher loading strategy

Both student and teacher are loaded fresh from the same model ID. This ensures:
1. Identical initial weights (EMA starts at identity).
2. No weight-sharing between student and teacher (shared weights would break the EMA update).
3. The teacher's parameters are immediately frozen (`teacher.freeze()`) before any eval to prevent accidental gradient accumulation.
