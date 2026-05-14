"""Minimal SDFT training example — ≤30 lines of user code.

Run:
    uv run python examples/minimal.py
"""

from mlx_sdft import SDFTTrainer, load_ultrachat

trainer = SDFTTrainer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    teacher_decay=0.99,
    lr=5e-6,
    grad_accum=2,
    max_new_tokens=64,
)

train_pairs, _ = load_ultrachat(
    trainer.tokenizer,
    n_train=200,
    n_eval=0,
    prompt_len=512,
    demo_len=128,
)

trainer.fit(train_pairs, steps=100, output_dir="outputs/minimal")
trainer.save("outputs/minimal/final")
print("Done. Weights saved to outputs/minimal/final/")
