"""Data utilities: generic pair tokeniser + dataset adapters.

A "pair" is a dict ``{"x_ids": list[int], "c_ids": list[int]}`` where
x_ids is the tokenised prompt and c_ids is the tokenised demonstration.

Usage::

    from mlx_sdft.data import tokenize_pairs, infinite_batches, load_ultrachat

    # Generic: supply your own list of {"prompt": str, "demonstration": str}
    pairs = tokenize_pairs(raw, tokenizer, prompt_len=1024, demo_len=256)

    # UltraChat shortcut:
    train, eval_ = load_ultrachat(tokenizer, n_train=5000, n_eval=500)

    # Infinite shuffled iterator for training:
    for batch in infinite_batches(pairs, seed=42):
        ...  # batch = {"x_ids": mx.array, "c_ids": mx.array}
"""

from __future__ import annotations

import random
from typing import Iterator

import mlx.core as mx
from tqdm import tqdm


def tokenize_pairs(
    data: list[dict],
    tokenizer,
    prompt_len: int,
    demo_len: int,
    seed: int = 42,
    prompt_key: str = "prompt",
    demonstration_key: str = "demonstration",
    desc: str = "Tokenising",
) -> list[dict]:
    """Tokenise a list of string-pair dicts into ``{x_ids, c_ids}`` dicts.

    data: list of ``{prompt_key: str, demonstration_key: str}``

    Prompts are right-truncated to ``prompt_len`` tokens; demonstrations are
    left-truncated to ``demo_len`` tokens. Pairs shorter than 4 tokens on
    either side are dropped.
    """
    rng = random.Random(seed)
    items = list(data)
    rng.shuffle(items)
    pairs: list[dict] = []
    skipped = 0
    for item in tqdm(items, desc=desc, unit="sample", dynamic_ncols=True):
        x_ids = tokenizer.encode(item[prompt_key])[-prompt_len:]
        c_ids = tokenizer.encode(item[demonstration_key])[:demo_len]
        if len(x_ids) < 4 or len(c_ids) < 4:
            skipped += 1
            continue
        pairs.append({"x_ids": x_ids, "c_ids": c_ids})
    if skipped:
        tqdm.write(f"  [{desc}] skipped {skipped} samples (too short)")
    return pairs


def load_ultrachat(
    tokenizer,
    n_train: int = 5000,
    n_eval: int = 500,
    prompt_len: int = 1024,
    demo_len: int = 256,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Download HuggingFaceH4/ultrachat_200k and return tokenised pair lists.

    Requires the ``datasets`` extra: ``uv add "mlx-sdft[data]"``.

    Returns ``(train_pairs, eval_pairs)`` where each pair is
    ``{"x_ids": list[int], "c_ids": list[int]}``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for load_ultrachat. "
            'Install with: uv add "mlx-sdft[data]"'
        ) from exc

    print("[data] Downloading HuggingFaceH4/ultrachat_200k …")
    raw_train = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    raw_eval = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")

    train_pairs = _build_pairs_from_messages(
        raw_train, tokenizer, prompt_len, demo_len, seed, "Tokenising train"
    )[:n_train]
    eval_pairs = _build_pairs_from_messages(
        raw_eval, tokenizer, prompt_len, demo_len, seed, "Tokenising eval"
    )[:n_eval]
    print(f"[data] {len(train_pairs)} train, {len(eval_pairs)} eval pairs ready")
    return train_pairs, eval_pairs


def _build_pairs_from_messages(
    dataset, tokenizer, prompt_len: int, demo_len: int, seed: int, desc: str
) -> list[dict]:
    """Internal: extract first user/assistant turns from a messages-style dataset."""
    rng = random.Random(seed)
    items = list(dataset)
    rng.shuffle(items)
    pairs: list[dict] = []
    skipped = 0
    for item in tqdm(items, desc=desc, unit="sample", dynamic_ncols=True):
        messages = item.get("messages") or item.get("conversations") or []
        if len(messages) < 2:
            skipped += 1
            continue
        user_turn = next((m for m in messages if m["role"] == "user"), None)
        asst_turn = next((m for m in messages if m["role"] == "assistant"), None)
        if user_turn is None or asst_turn is None:
            skipped += 1
            continue
        x_text = tokenizer.apply_chat_template(
            [user_turn], tokenize=False, add_generation_prompt=True
        )
        x_ids = tokenizer.encode(x_text)[-prompt_len:]
        c_ids = tokenizer.encode(asst_turn["content"])[:demo_len]
        if len(x_ids) < 4 or len(c_ids) < 4:
            skipped += 1
            continue
        pairs.append({"x_ids": x_ids, "c_ids": c_ids})
    if skipped:
        tqdm.write(f"  [{desc}] skipped {skipped} samples")
    return pairs


def infinite_batches(
    pairs: list[dict],
    seed: int = 42,
) -> Iterator[dict[str, mx.array]]:
    """Yield single-sample mx.array batches indefinitely, shuffling each epoch.

    Each yielded batch: ``{"x_ids": mx.array([...], int32), "c_ids": ...}``
    """
    rng = random.Random(seed)
    while True:
        epoch = list(pairs)
        rng.shuffle(epoch)
        for item in epoch:
            yield {
                "x_ids": mx.array(item["x_ids"], dtype=mx.int32),
                "c_ids": mx.array(item["c_ids"], dtype=mx.int32),
            }
