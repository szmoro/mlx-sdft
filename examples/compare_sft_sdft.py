"""Side-by-side SFT vs SDFT training + KL-from-base comparison plot.

Trains both methods on the same data for the same number of steps and
plots the loss curves and KL-from-base divergence.

Run:
    uv run python examples/compare_sft_sdft.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

from mlx_sdft import SDFTTrainer, load_ultrachat, sft_loss
from mlx_sdft.data import infinite_batches


STEPS = 200
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN = 1000
OUTPUT_DIR = Path("outputs/compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_sft(train_pairs: list[dict]) -> list[dict]:
    print("\n=== SFT baseline ===")
    student, tokenizer = load(MODEL_ID)
    student.train()
    optimizer = optim.AdamW(learning_rate=5e-6)

    def loss_fn(model, batch):
        return sft_loss(model, batch["x_ids"], batch["c_ids"])

    vg = nn.value_and_grad(student, loss_fn)
    batch_iter = infinite_batches(train_pairs)
    records: list[dict] = []
    for step in range(1, STEPS + 1):
        batch = next(batch_iter)
        (loss, grads) = vg(student, batch)
        mx.eval(loss, grads)
        optimizer.update(student, grads)
        mx.eval(student.parameters())
        if step % 20 == 0:
            records.append({"step": step, "loss": float(loss.item())})
            print(f"  SFT step {step:3d}  loss={loss.item():.4f}")

    # Save weights for KL eval
    flat = dict(tree_flatten(student.parameters()))
    mx.savez(str(OUTPUT_DIR / "sft_weights.npz"), **flat)
    return records


def run_sdft(train_pairs: list[dict]) -> list[dict]:
    print("\n=== SDFT ===")
    trainer = SDFTTrainer.from_pretrained(
        MODEL_ID, teacher_decay=0.99, lr=5e-6, grad_accum=2, max_new_tokens=64
    )
    metrics = trainer.fit(train_pairs, steps=STEPS, log_every=20, output_dir=OUTPUT_DIR / "sdft")
    trainer.save(OUTPUT_DIR / "sdft_final")
    return metrics


def kl_from_base(weights_path: str, model_id: str, prompts: list[list[int]]) -> float:
    """Mean token-level KL between fine-tuned model and base model on prompts."""
    base, _ = load(model_id)
    finetuned, _ = load(model_id)
    saved = dict(mx.load(weights_path))
    from mlx.utils import tree_unflatten
    finetuned.update(tree_unflatten(list(saved.items())))
    mx.eval(finetuned.parameters())

    kls = []
    for ids in prompts[:50]:
        x = mx.array(ids[:128], dtype=mx.int32)[None]  # [1, T]
        logits_base = base(x)[0]
        logits_ft = finetuned(x)[0]
        p_base = nn.softmax(logits_base, axis=-1)
        log_p_ft = nn.log_softmax(logits_ft, axis=-1)
        kl = mx.sum(p_base * (mx.log(p_base + 1e-9) - log_p_ft), axis=-1)
        mx.eval(kl)
        kls.append(float(mx.mean(kl).item()))
    return float(np.mean(kls))


def plot(sft_records: list[dict], sdft_records: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot([r["step"] for r in sft_records], [r["loss"] for r in sft_records], label="SFT")
    ax.plot([r["step"] for r in sdft_records], [r["loss"] for r in sdft_records], label="SDFT")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()

    ax = axes[1]
    ax.set_title("KL-from-base (lower = less forgetting)")
    ax.set_ylabel("Mean token KL (nats)")
    ax.set_xticks([])

    fig.tight_layout()
    out = OUTPUT_DIR / "compare_sft_sdft.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


def main():
    print(f"Loading {N_TRAIN} UltraChat pairs for {MODEL_ID} …")
    _, tokenizer = load(MODEL_ID)
    train_pairs, _ = load_ultrachat(tokenizer, n_train=N_TRAIN, n_eval=0, prompt_len=512, demo_len=128)

    sft_records = run_sft(train_pairs)
    sdft_records = run_sdft(train_pairs)

    (OUTPUT_DIR / "sft_metrics.json").write_text(json.dumps(sft_records, indent=2))
    (OUTPUT_DIR / "sdft_metrics.json").write_text(json.dumps(sdft_records, indent=2))

    plot(sft_records, sdft_records)
    print("\nDone.")


if __name__ == "__main__":
    main()
