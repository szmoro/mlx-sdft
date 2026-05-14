"""Full SDFT training recipe for Qwen2.5-1.5B on UltraChat-200k.

Matches the hyperparameters from the mlx-sdft research experiment:
  Model   : Qwen/Qwen2.5-1.5B-Instruct
  Dataset : HuggingFaceH4/ultrachat_200k (5k train / 500 eval)
  Steps   : 500, lr=5e-6, grad_accum=4, EMA=0.999

Expected wall-clock time on M1 Pro 32 GB: ~2 hours.

Run:
    uv run python examples/train_qwen.py [--steps 500] [--output outputs/qwen]
"""

import argparse
from pathlib import Path

from mlx_sdft import SDFTTrainer, load_ultrachat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", default="outputs/qwen")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="Push to HuggingFace Hub after training, e.g. szmoro/my-model")
    args = parser.parse_args()

    trainer = SDFTTrainer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        teacher_decay=0.999,
        lr=5e-6,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        grad_accum=4,
        grad_clip=1.0,
        max_new_tokens=256,
        sampler_temp=1.0,
        sampler_top_p=0.9,
    )

    train_pairs, _ = load_ultrachat(
        trainer.tokenizer,
        n_train=5000,
        n_eval=0,
        prompt_len=1024,
        demo_len=256,
    )

    trainer.fit(
        train_pairs,
        steps=args.steps,
        log_every=10,
        save_every=250,
        output_dir=args.output,
    )

    out = Path(args.output) / "final"
    trainer.save(out)
    print(f"Weights saved to {out}/")

    if args.push_to_hub:
        url = trainer.push_to_hub(args.push_to_hub)
        print(f"Pushed to {url}")


if __name__ == "__main__":
    main()
