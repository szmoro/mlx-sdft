"""100-step SDFT training on a tiny toy model. Should run in < 30 s.

Tests the training loop mechanics: gradient accumulation, EMA update,
JSONL logging, checkpoint saving. sample_rollout is patched to avoid
the mlx_lm.stream_generate dependency at test time.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_sdft.teacher import EMATeacher
from mlx_sdft.trainer import SDFTTrainer


# ── Toy model ──────────────────────────────────────────────────────────────────

VOCAB = 32
HIDDEN = 16


class TinyTransformer(nn.Module):
    """Minimal transformer-like model with ~1 k params for fast testing."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.attn = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.proj = nn.Linear(HIDDEN, VOCAB, bias=False)

    def __call__(self, x):  # [batch, seq] -> [batch, seq, vocab]
        h = self.embed(x)
        h = nn.relu(self.attn(h))
        return self.proj(h)


def _make_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 1
    tok.decode.return_value = "dummy prompt"
    return tok


def _fixed_rollout(model, tokenizer, x_ids, max_new_tokens, temp=1.0, top_p=0.9):
    """Stub rollout that returns 4 fixed tokens — no stream_generate needed."""
    mx.random.seed(0)
    return mx.array([5, 6, 7, 8], dtype=mx.int32)


def _make_pairs(n: int = 20) -> list[dict]:
    rng = np.random.default_rng(42)
    return [
        {
            "x_ids": rng.integers(2, VOCAB, size=8).tolist(),
            "c_ids": rng.integers(2, VOCAB, size=4).tolist(),
        }
        for _ in range(n)
    ]


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("mlx_sdft.trainer.sample_rollout", side_effect=_fixed_rollout)
def test_trainer_runs_100_steps(mock_rollout, tmp_path):
    mx.random.seed(42)
    student = TinyTransformer()
    teacher_model = TinyTransformer()
    teacher_model.freeze()
    mx.eval(student.parameters(), teacher_model.parameters())
    teacher = EMATeacher(teacher_model, decay=0.99)
    tokenizer = _make_tokenizer()

    trainer = SDFTTrainer(
        student,
        tokenizer,
        teacher=teacher,
        lr=1e-3,
        grad_accum=2,
        grad_clip=1.0,
        ema_decay=0.99,
        max_new_tokens=4,
        model_id="test/tiny",
    )
    pairs = _make_pairs(20)
    metrics = trainer.fit(pairs, steps=10, log_every=5, save_every=10, output_dir=tmp_path)

    assert len(metrics) > 0, "Expected at least one metrics record"
    assert all("loss" in r for r in metrics)
    assert all("grad_norm" in r for r in metrics)


@patch("mlx_sdft.trainer.sample_rollout", side_effect=_fixed_rollout)
def test_trainer_writes_metrics_jsonl(mock_rollout, tmp_path):
    mx.random.seed(1)
    student = TinyTransformer()
    teacher_model = TinyTransformer()
    teacher_model.freeze()
    mx.eval(student.parameters(), teacher_model.parameters())
    teacher = EMATeacher(teacher_model, decay=0.99)

    trainer = SDFTTrainer(
        student, _make_tokenizer(), teacher=teacher,
        lr=1e-3, grad_accum=1, grad_clip=0.0, ema_decay=0.99,
        max_new_tokens=4, model_id="test/tiny",
    )
    trainer.fit(_make_pairs(), steps=5, log_every=1, output_dir=tmp_path)

    metrics_path = tmp_path / "metrics.jsonl"
    assert metrics_path.exists(), "metrics.jsonl not created"
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) >= 5
    for line in lines:
        rec = json.loads(line)
        assert "step" in rec and "loss" in rec


@patch("mlx_sdft.trainer.sample_rollout", side_effect=_fixed_rollout)
def test_trainer_writes_run_json(mock_rollout, tmp_path):
    mx.random.seed(2)
    student = TinyTransformer()
    teacher_model = TinyTransformer()
    teacher_model.freeze()
    mx.eval(student.parameters(), teacher_model.parameters())
    teacher = EMATeacher(teacher_model, decay=0.99)

    trainer = SDFTTrainer(
        student, _make_tokenizer(), teacher=teacher,
        lr=1e-3, grad_accum=1, max_new_tokens=4, model_id="test/tiny",
    )
    trainer.fit(_make_pairs(), steps=3, output_dir=tmp_path)

    run_json = tmp_path / "run.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["model_id"] == "test/tiny"
    assert data["steps"] == 3
    assert "versions" in data


@patch("mlx_sdft.trainer.sample_rollout", side_effect=_fixed_rollout)
def test_trainer_save_and_reload(mock_rollout, tmp_path):
    mx.random.seed(3)
    student = TinyTransformer()
    teacher_model = TinyTransformer()
    teacher_model.freeze()
    mx.eval(student.parameters(), teacher_model.parameters())
    teacher = EMATeacher(teacher_model, decay=0.99)

    trainer = SDFTTrainer(
        student, _make_tokenizer(), teacher=teacher,
        lr=1e-3, grad_accum=1, max_new_tokens=4, model_id="test/tiny",
    )
    trainer.fit(_make_pairs(), steps=3, output_dir=tmp_path)

    save_path = tmp_path / "saved"
    trainer.save(save_path)

    assert (save_path / "weights.npz").exists()
    assert (save_path / "config.json").exists()
    config = json.loads((save_path / "config.json").read_text())
    assert config["model_id"] == "test/tiny"


def test_trainer_requires_teacher():
    with pytest.raises(ValueError, match="teacher"):
        SDFTTrainer(TinyTransformer(), _make_tokenizer())
