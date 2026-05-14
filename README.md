# mlx-sdft

**Self-Distillation Fine-Tuning for Apple Silicon** — MLX implementation of
[Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897).

SDFT fine-tunes language models on new tasks while preserving prior capabilities,
by distilling the model from its own EMA-smoothed, instruction-conditioned version.
It achieves significantly lower KL-from-base than standard SFT (0.68 vs 1.26 nats
in the paper) without sacrificing new-task performance.

---

## Install

```bash
uv add mlx-sdft
# or, for data loading helpers:
uv add "mlx-sdft[data]"
```

Requires macOS with Apple Silicon (MLX). Python 3.14+.

---

## Quickstart

```python
from mlx_sdft import SDFTTrainer, load_ultrachat

trainer = SDFTTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    teacher_decay=0.999,
    lr=5e-6,
    grad_accum=4,
    max_new_tokens=256,
)

train_pairs, _ = load_ultrachat(trainer.tokenizer, n_train=5000, n_eval=500)

trainer.fit(train_pairs, steps=500)
trainer.save("./checkpoints/run1")
trainer.push_to_hub("szmoro/my-sdft-model")
```

Full script: [`examples/train_qwen.py`](examples/train_qwen.py)

---

## Why SDFT?

Standard SFT updates all parameters toward a new task and forgets prior knowledge.
SDFT uses the model as its own teacher:

- **Student** sees only the prompt and receives gradient.
- **Teacher** is an EMA copy that also sees the demonstration, acting as a
  soft-distillation target.

The reverse-KL loss on student rollouts keeps the student from drifting far from
its base policy, with the demonstration providing the instruction signal through
the teacher.

```
L_SDFT = E_{y ~ π_student(·|x)} [ log π_student(y|x) − log π_teacher(y|x,c) ]
```

See [`docs/algorithm.md`](docs/algorithm.md) for the full derivation and data-flow diagram.

---

## API

### High-level (recommended)

| Method | Description |
|--------|-------------|
| `SDFTTrainer.from_pretrained(model_id, **hp)` | Load student + teacher, return trainer |
| `trainer.fit(pairs, steps, ...)` | Run SDFT training loop |
| `trainer.save(path)` | Save weights + config |
| `trainer.push_to_hub(repo_id)` | Upload to HuggingFace Hub |

### Functional (for custom loops and subclassing)

| Symbol | Description |
|--------|-------------|
| `sdft_loss(student, teacher, x, c, y)` | Reverse-KL loss |
| `sft_loss(student, x, c)` | Standard NLL loss |
| `sample_rollout(model, tokenizer, x, n)` | Detached student rollout |
| `EMATeacher(model, decay)` | Frozen EMA teacher wrapper |
| `load_ultrachat(tokenizer, ...)` | Download + tokenise UltraChat-200k |
| `tokenize_pairs(data, tokenizer, ...)` | Generic string-pair tokeniser |
| `infinite_batches(pairs, seed)` | Infinite shuffled batch iterator |

### Subclassing for custom objectives

Override `_make_loss_fn` to inject a different loss:

```python
class CustomSDFTTrainer(SDFTTrainer):
    def _make_loss_fn(self):
        base_fn = super()._make_loss_fn()
        def loss_fn(model, batch):
            # modify or wrap the base loss
            ...
        return loss_fn
```

---

## Examples

| Script | Description |
|--------|-------------|
| [`examples/minimal.py`](examples/minimal.py) | 30-line end-to-end on 0.5B |
| [`examples/train_qwen.py`](examples/train_qwen.py) | Full 1.5B / UltraChat recipe |
| [`examples/compare_sft_sdft.py`](examples/compare_sft_sdft.py) | Side-by-side loss + KL plot |

---

## Citation

```bibtex
@article{shenfeld2026sdft,
  title={Self-Distillation Fine-Tuning for Continual Learning in Language Models},
  author={Shenfeld, Idan and others},
  year={2026},
  url={https://arxiv.org/abs/2601.19897}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
