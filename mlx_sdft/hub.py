"""HuggingFace Hub utilities: push weights and generate model cards."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


def push_to_hub(
    weights: dict,
    config: dict,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    model_card: str | None = None,
) -> str:
    """Push model weights and config to HuggingFace Hub.

    weights   : flat param dict from ``dict(tree_flatten(model.parameters()))``
    config    : training config dict (model_id, hyperparams, metrics)
    repo_id   : ``"username/repo-name"`` on HuggingFace
    token     : HF token; if None uses ``HF_TOKEN`` env var or cached login
    private   : create as private repo
    model_card: README.md content; if None, one is auto-generated

    Returns the repo URL.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required. Install with: uv add mlx-sdft"
        ) from exc

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Weights
        weights_path = tmp_path / "weights.npz"
        mx.savez(str(weights_path), **weights)
        api.upload_file(
            path_or_fileobj=str(weights_path),
            path_in_repo="weights.npz",
            repo_id=repo_id,
            token=token,
        )

        # Config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config, indent=2))
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
            repo_id=repo_id,
            token=token,
        )

        # Model card
        if model_card is None:
            model_card = generate_model_card(
                config.get("model_id", "unknown"),
                repo_id,
                config,
            )
        readme_path = tmp_path / "README.md"
        readme_path.write_text(model_card)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )

    return f"https://huggingface.co/{repo_id}"


def generate_model_card(
    base_model_id: str,
    repo_id: str,
    config: dict,
    metrics: dict | None = None,
) -> str:
    """Generate a minimal HuggingFace model card for an SDFT fine-tune."""
    steps = config.get("steps", "?")
    lr = config.get("lr", "?")
    ema_decay = config.get("ema_decay", "?")
    dataset = config.get("dataset", "HuggingFaceH4/ultrachat_200k")

    metrics_block = ""
    if metrics:
        rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
        metrics_block = f"""
## Evaluation

| Metric | Value |
|--------|-------|
{rows}
"""

    return f"""---
base_model: {base_model_id}
library_name: mlx-sdft
tags:
  - mlx
  - sdft
  - continual-learning
  - apple-silicon
license: apache-2.0
---

# {repo_id.split("/")[-1]}

Fine-tuned from [{base_model_id}](https://huggingface.co/{base_model_id}) using
**Self-Distillation Fine-Tuning (SDFT)** ([Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897))
via [mlx-sdft](https://github.com/szmoro/mlx-sdft) on Apple Silicon.

SDFT reduces catastrophic forgetting by having the model act as its own teacher:
the student (prompt-only) is distilled from the EMA-teacher (prompt + demonstration)
via a reverse-KL loss on student rollouts.

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | `{base_model_id}` |
| Dataset | `{dataset}` |
| Steps | {steps} |
| Learning rate | {lr} |
| EMA decay | {ema_decay} |
{metrics_block}
## Usage

```python
from mlx_lm import load, generate
model, tokenizer = load("{repo_id}")
print(generate(model, tokenizer, prompt="Hello!", max_tokens=200))
```

## Citation

```bibtex
@article{{shenfeld2026sdft,
  title={{Self-Distillation Fine-Tuning for Continual Learning in Language Models}},
  author={{Shenfeld, Idan and others}},
  year={{2026}},
  url={{https://arxiv.org/abs/2601.19897}}
}}
```
"""
