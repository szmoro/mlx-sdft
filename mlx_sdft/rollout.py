"""Student rollout sampling.

Wraps mlx_lm.stream_generate to produce a detached token sequence
y ~ π_student(·|x). No gradient flows through the sampling step —
gradients are computed separately via teacher-forced forwards in losses.py.

Usage::

    from mlx_sdft import sample_rollout

    y_ids = sample_rollout(student, tokenizer, x_ids, max_new_tokens=256)
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler


def sample_rollout(
    student,
    tokenizer,
    x_ids: mx.array,
    max_new_tokens: int,
    temp: float = 1.0,
    top_p: float = 0.9,
) -> mx.array:
    """Sample y ~ π_student(·|x). Returns int32 token ids; no gradient.

    x_ids         : prompt token ids [L_x]
    max_new_tokens: hard cap on rollout length
    temp          : sampling temperature (1.0 = no sharpening)
    top_p         : nucleus sampling threshold

    Returns shape [T] where T ≤ max_new_tokens.
    """
    prompt_text = tokenizer.decode(x_ids.tolist())
    sampler = make_sampler(temp=temp, top_p=top_p)
    y_tokens: list[int] = []
    for response in stream_generate(
        student,
        tokenizer,
        prompt_text,
        max_tokens=max_new_tokens,
        sampler=sampler,
    ):
        tok = response.token
        if tok == tokenizer.eos_token_id:
            break
        y_tokens.append(tok)
        if len(y_tokens) >= max_new_tokens:
            break
    if not y_tokens:
        y_tokens = [tokenizer.eos_token_id]
    return mx.array(y_tokens, dtype=mx.int32)
