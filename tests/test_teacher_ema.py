"""EMA teacher math and dtype preservation tests."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_sdft.teacher import EMATeacher


class Tiny(nn.Module):
    def __init__(self, val: float = 1.0):
        super().__init__()
        self.w = mx.array([[val]], dtype=mx.float32)

    def __call__(self, x):
        return x @ self.w


def test_ema_single_update():
    """After 1 update: teacher = α·t0 + (1−α)·s"""
    alpha = 0.9
    teacher_model = Tiny(val=2.0)
    student = Tiny(val=4.0)
    mx.eval(teacher_model.parameters(), student.parameters())

    teacher = EMATeacher(teacher_model, decay=alpha)
    teacher.update(student)

    expected = alpha * 2.0 + (1 - alpha) * 4.0  # 0.9*2 + 0.1*4 = 2.2
    actual = float(teacher.model.w.item())
    assert abs(actual - expected) < 1e-5, f"{actual} != {expected}"


def test_ema_n_updates_converges():
    """After N updates of a constant student, teacher → student value."""
    alpha = 0.9
    N = 200
    t0, s_val = 0.0, 1.0
    teacher_model = Tiny(val=t0)
    student = Tiny(val=s_val)
    mx.eval(teacher_model.parameters(), student.parameters())

    teacher = EMATeacher(teacher_model, decay=alpha)
    for _ in range(N):
        teacher.update(student)

    # Closed form: t_N = α^N * t0 + (1 - α^N) * s
    expected = (alpha**N) * t0 + (1 - alpha**N) * s_val
    actual = float(teacher.model.w.item())
    assert abs(actual - expected) < 1e-4, f"{actual} != {expected}"


def test_ema_dtype_preserved():
    """Teacher parameters should keep their original dtype after update."""
    mx.random.seed(0)
    teacher_model = Tiny()
    student = Tiny()
    # Cast to bfloat16
    from mlx.utils import tree_flatten, tree_unflatten
    t_flat = [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(teacher_model.parameters())]
    teacher_model.update(tree_unflatten(t_flat))
    s_flat = [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(student.parameters())]
    student.update(tree_unflatten(s_flat))
    mx.eval(teacher_model.parameters(), student.parameters())

    teacher = EMATeacher(teacher_model, decay=0.99)
    teacher.update(student)

    for _, v in tree_flatten(teacher.model.parameters()):
        assert v.dtype == mx.bfloat16, f"Expected bfloat16, got {v.dtype}"


def test_ema_invalid_decay():
    model = Tiny()
    with pytest.raises(ValueError, match="decay"):
        EMATeacher(model, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        EMATeacher(model, decay=0.0)


def test_ema_save(tmp_path):
    """save() should produce a readable .npz file."""
    teacher_model = Tiny(val=3.5)
    mx.eval(teacher_model.parameters())
    teacher = EMATeacher(teacher_model)
    out = tmp_path / "teacher.npz"
    teacher.save(out)
    data = mx.load(str(out))
    assert "w" in data
    assert abs(float(data["w"].item()) - 3.5) < 1e-5
