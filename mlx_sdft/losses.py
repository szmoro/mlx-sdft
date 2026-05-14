"""SDFT and SFT loss functions.

SDFT reverse-KL loss (Shenfeld et al. 2026, https://arxiv.org/abs/2601.19897):

    L_SDFT = E_{y ~ π_student(·|x)} [ log π_student(y|x) − log π_teacher(y|x,c) ]

The expectation is approximated by a single rollout y sampled from π_student.
Gradient flows only through the student log-probs; teacher is called under
mx.stop_gradient to ensure it contributes no gradient to the student.

Usage::

    from mlx_sdft import sdft_loss, sft_loss

    loss = sdft_loss(student, teacher.model, x_ids, c_ids, y_ids)
    loss = sft_loss(student, x_ids, c_ids)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _logprobs_at_positions(
    model: nn.Module,
    input_ids: mx.array,
    target_ids: mx.array,
) -> mx.array:
    """Teacher-forced forward pass; returns per-token log-probs for target_ids.

    input_ids  : shape [seq_len]  — full context (prompt + optional demo + target)
    target_ids : shape [T]        — the last T tokens whose log-probs to return

    Returns shape [T].
    """
    logits = model(input_ids[None])[0]      # [seq_len, vocab]
    T = target_ids.shape[0]
    pred_logits = logits[-(T + 1) : -1]     # [T, vocab] — left-shift by 1
    log_probs_all = nn.log_softmax(pred_logits, axis=-1)
    return log_probs_all[mx.arange(T), target_ids]   # [T]


def sdft_loss(
    student: nn.Module,
    teacher: nn.Module,
    x_ids: mx.array,
    c_ids: mx.array,
    y_ids: mx.array,
) -> mx.array:
    """Reverse-KL SDFT loss.

    student  — receives gradient
    teacher  — frozen EMA copy; wrapped in mx.stop_gradient
    x_ids    — prompt tokens [L_x]
    c_ids    — demonstration tokens [L_c]
    y_ids    — student rollout tokens [T] (sampled, no grad through sampling)

    Student context:  x_ids ++ y_ids
    Teacher context:  x_ids ++ c_ids ++ y_ids
    """
    student_input = mx.concatenate([x_ids, y_ids], axis=0)
    log_p_student = _logprobs_at_positions(student, student_input, y_ids)

    teacher_input = mx.concatenate([x_ids, c_ids, y_ids], axis=0)
    log_p_teacher = mx.stop_gradient(
        _logprobs_at_positions(teacher, teacher_input, y_ids)
    )

    return mx.mean(log_p_student - log_p_teacher)


def sft_loss(
    student: nn.Module,
    x_ids: mx.array,
    c_ids: mx.array,
) -> mx.array:
    """Standard NLL (cross-entropy) SFT loss: −E[log π(c|x)]."""
    input_ids = mx.concatenate([x_ids, c_ids], axis=0)
    logits = student(input_ids[None])[0]    # [seq, vocab]
    T = c_ids.shape[0]
    pred_logits = logits[-(T + 1) : -1]
    log_probs = nn.log_softmax(pred_logits, axis=-1)
    nll = -log_probs[mx.arange(T), c_ids]
    return mx.mean(nll)
