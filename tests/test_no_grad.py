"""Tests for mojodl.no_grad() and mojodl.is_grad_enabled().

PyTorch semantics: inside no_grad, op results have requires_grad False,
grad_fn None, is_leaf True, and no tensors are saved for backward. On exit
(also via an exception) the previous mode is restored; nesting works.

Memory counters are process-wide: gc.collect() then take a baseline, assert
deltas. Layout uses N = 64*64*4 and f(x, w) = (((x * w) * w) * w).sum().
"""

import gc

import numpy as np
import pytest

import mojodl

SHAPE = (64, 64)
N = 64 * 64 * 4  # 16384


def _random_xw(seed=0):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(SHAPE).astype(np.float32)
    w_np = rng.standard_normal(SHAPE).astype(np.float32)
    return x_np, w_np


def _f(x, w):
    return (((x * w) * w) * w).sum()


def test_no_grad_results_do_not_track_and_match_values_outside():
    """Under no_grad, mul/add/f drop history; values match the same ops outside."""
    x_np, w_np = _random_xw()
    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np, requires_grad=True)

    with_grad_mul = (x * w).numpy()
    with_grad_add = (x + w).numpy()
    with_grad_f = _f(x, w).numpy()

    with mojodl.no_grad():
        mul = x * w
        add = x + w
        loss = _f(x, w)

    for t in (mul, add, loss):
        assert t.requires_grad is False
        assert t.grad_fn is None
        assert t.is_leaf is True

    np.testing.assert_allclose(mul.numpy(), with_grad_mul, rtol=0, atol=0)
    np.testing.assert_allclose(add.numpy(), with_grad_add, rtol=0, atol=0)
    np.testing.assert_allclose(loss.numpy(), with_grad_f, rtol=0, atol=0)


def test_is_grad_enabled_default_inside_nested_and_after_exit():
    """is_grad_enabled is True by default, False in (nested) no_grad, True after."""
    assert mojodl.is_grad_enabled() is True

    with mojodl.no_grad():
        assert mojodl.is_grad_enabled() is False
        with mojodl.no_grad():
            assert mojodl.is_grad_enabled() is False
        assert mojodl.is_grad_enabled() is False

    assert mojodl.is_grad_enabled() is True


def test_is_grad_enabled_restored_after_exception_inside_no_grad():
    """Exception leaving no_grad still restores grad mode."""
    assert mojodl.is_grad_enabled() is True

    with pytest.raises(RuntimeError, match="boom"):
        with mojodl.no_grad():
            assert mojodl.is_grad_enabled() is False
            raise RuntimeError("boom")

    assert mojodl.is_grad_enabled() is True


def test_after_no_grad_ops_record_history_again():
    """Leaving no_grad restores recording: x * w gets a grad_fn."""
    x_np, w_np = _random_xw()
    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np, requires_grad=True)

    with mojodl.no_grad():
        inside = x * w
    assert inside.grad_fn is None

    outside = x * w
    assert outside.grad_fn is not None
    assert outside.requires_grad is True
    assert outside.is_leaf is False


def test_no_grad_saves_nothing_only_loss_bytes():
    """With x, w alive, no_grad f keeps only the scalar loss (+4)."""
    x_np, w_np = _random_xw()
    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np, requires_grad=True)

    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    with mojodl.no_grad():
        loss = _f(x, w)

    assert mojodl.memory_stats()["current_bytes"] - baseline == 4
    del loss
    gc.collect()


def test_with_grad_keeps_two_saved_intermediates():
    """Without no_grad, f keeps two saved intermediates (2*N + 4) beyond x, w."""
    x_np, w_np = _random_xw()
    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np, requires_grad=True)

    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    loss = _f(x, w)

    assert mojodl.memory_stats()["current_bytes"] - baseline == 2 * N + 4
    del loss
    gc.collect()


def test_no_grad_peak_strictly_lower_than_with_grad():
    """Peak delta for f under no_grad is strictly smaller than with grad enabled."""
    x_np, w_np = _random_xw()
    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np, requires_grad=True)

    gc.collect()
    mojodl.reset_peak_memory_stats()
    base = mojodl.memory_stats()["current_bytes"]

    with mojodl.no_grad():
        loss = _f(x, w)
    no_grad_peak_delta = mojodl.memory_stats()["peak_bytes"] - base

    del loss
    gc.collect()
    mojodl.reset_peak_memory_stats()
    base = mojodl.memory_stats()["current_bytes"]

    loss = _f(x, w)
    with_grad_peak_delta = mojodl.memory_stats()["peak_bytes"] - base

    assert no_grad_peak_delta < with_grad_peak_delta
    del loss
    gc.collect()
