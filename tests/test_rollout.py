"""Shape, dtype, determinism, and EOS-termination tests for sample_rollout."""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from mlx_sdft.rollout import sample_rollout


def _make_mock_tokenizer(vocab_size: int = 100, eos_id: int = 2):
    tok = MagicMock()
    tok.eos_token_id = eos_id
    tok.decode.return_value = "hello world"
    return tok


def _stream_response(token_ids: list[int]):
    """Yield MagicMock responses for stream_generate."""
    for tid in token_ids:
        r = MagicMock()
        r.token = tid
        yield r


@patch("mlx_sdft.rollout.stream_generate")
@patch("mlx_sdft.rollout.make_sampler")
def test_output_dtype(mock_sampler, mock_stream):
    mock_sampler.return_value = MagicMock()
    mock_stream.return_value = iter(_stream_response([5, 6, 7]))
    tok = _make_mock_tokenizer()
    student = MagicMock()
    x_ids = mx.array([1, 2, 3], dtype=mx.int32)

    y = sample_rollout(student, tok, x_ids, max_new_tokens=10)
    assert y.dtype == mx.int32, f"Expected int32, got {y.dtype}"


@patch("mlx_sdft.rollout.stream_generate")
@patch("mlx_sdft.rollout.make_sampler")
def test_output_length_bounded(mock_sampler, mock_stream):
    """Rollout should not exceed max_new_tokens."""
    MAX = 5
    mock_sampler.return_value = MagicMock()
    # Provide more tokens than the cap
    mock_stream.return_value = iter(_stream_response(list(range(10, 30))))
    tok = _make_mock_tokenizer()
    student = MagicMock()
    x_ids = mx.array([1, 2], dtype=mx.int32)

    y = sample_rollout(student, tok, x_ids, max_new_tokens=MAX)
    assert y.shape[0] <= MAX, f"Expected ≤{MAX} tokens, got {y.shape[0]}"


@patch("mlx_sdft.rollout.stream_generate")
@patch("mlx_sdft.rollout.make_sampler")
def test_eos_terminates_early(mock_sampler, mock_stream):
    """EOS token should stop rollout before max_new_tokens."""
    eos = 2
    mock_sampler.return_value = MagicMock()
    # tokens: 5, EOS (should stop at 1 token)
    mock_stream.return_value = iter(_stream_response([5, eos, 6, 7]))
    tok = _make_mock_tokenizer(eos_id=eos)
    student = MagicMock()
    x_ids = mx.array([1], dtype=mx.int32)

    y = sample_rollout(student, tok, x_ids, max_new_tokens=10)
    assert y.shape[0] == 1, f"EOS should stop after 1 token, got {y.shape[0]}"


@patch("mlx_sdft.rollout.stream_generate")
@patch("mlx_sdft.rollout.make_sampler")
def test_empty_stream_returns_eos(mock_sampler, mock_stream):
    """Empty generation should return a single EOS token, not an empty array."""
    eos = 2
    mock_sampler.return_value = MagicMock()
    mock_stream.return_value = iter([])
    tok = _make_mock_tokenizer(eos_id=eos)
    student = MagicMock()
    x_ids = mx.array([1, 2], dtype=mx.int32)

    y = sample_rollout(student, tok, x_ids, max_new_tokens=10)
    assert y.shape[0] == 1
    assert int(y[0].item()) == eos


@patch("mlx_sdft.rollout.stream_generate")
@patch("mlx_sdft.rollout.make_sampler")
def test_deterministic_with_fixed_mock(mock_sampler, mock_stream):
    """Two calls with the same mock should return identical results."""
    tokens = [10, 20, 30]
    tok = _make_mock_tokenizer()
    student = MagicMock()
    x_ids = mx.array([1, 2], dtype=mx.int32)

    mock_sampler.return_value = MagicMock()
    mock_stream.return_value = iter(_stream_response(tokens))
    y1 = sample_rollout(student, tok, x_ids, max_new_tokens=20)

    mock_stream.return_value = iter(_stream_response(tokens))
    y2 = sample_rollout(student, tok, x_ids, max_new_tokens=20)

    assert y1.tolist() == y2.tolist()
