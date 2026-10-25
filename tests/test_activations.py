"""Tests for Tensor.relu / mojodl.relu forward, backward, and Meta behaviour.

PyTorch (CPU) is the numerical oracle. ReLU is exact on float32, so
comparisons use rtol=0 / atol=0.
"""

import gc

import numpy as np
import pytest
import torch

import mojodl

RELU_RTOL = 0
RELU_ATOL = 0

_FORWARD_SHAPES = [
    (),
    (7,),
    (3, 4),
    (2, 3, 5),
    (0, 3),
]


def _seeded_with_zeros(shape, seed=0):
    """Standard-normal float32 input with exact 0.0 entries where shape allows."""
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(shape).astype(np.float32)
    if x_np.size == 0:
        return x_np
    flat = x_np.reshape(-1)
    # Plant exact zeros so the ReLU kink is covered (grad must be 0 there).
    n_zeros = max(1, flat.size // 4)
    idxs = rng.choice(flat.size, size=n_zeros, replace=False)
    flat[idxs] = 0.0
    return x_np


@pytest.mark.parametrize("shape", _FORWARD_SHAPES, ids=["scalar", "1d", "2d", "3d", "empty"])
def test_relu_method_forward_matches_torch(shape):
    x_np = _seeded_with_zeros(shape)
    x = mojodl.Tensor(x_np)

    result = x.relu()
    expected = torch.relu(torch.from_numpy(x_np)).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=RELU_RTOL, atol=RELU_ATOL
    )


@pytest.mark.parametrize("shape", _FORWARD_SHAPES, ids=["scalar", "1d", "2d", "3d", "empty"])
def test_relu_function_forward_matches_torch(shape):
    x_np = _seeded_with_zeros(shape)
    x = mojodl.Tensor(x_np)

    result = mojodl.relu(x)
    expected = torch.relu(torch.from_numpy(x_np)).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=RELU_RTOL, atol=RELU_ATOL
    )


def test_relu_does_not_mutate_input():
    x_np = _seeded_with_zeros((3, 4))
    x = mojodl.Tensor(x_np)

    _ = x.relu()
    _ = mojodl.relu(x)

    np.testing.assert_array_equal(x.numpy(), x_np)


def test_relu_result_storage_is_accounted():
    shape = (3, 4)
    x_np = _seeded_with_zeros(shape)
    x = mojodl.Tensor(x_np)
    nbytes = int(np.prod(shape)) * 4
    baseline = mojodl.memory_stats()["current_bytes"]

    result = x.relu()

    assert mojodl.memory_stats()["current_bytes"] - baseline == nbytes
    del result
    gc.collect()


@pytest.mark.parametrize(
    "shape",
    [(), (7,), (3, 4), (2, 3, 5)],
    ids=["scalar", "1d", "2d", "3d"],
)
def test_relu_backward_matches_torch(shape):
    x_np = _seeded_with_zeros(shape, seed=0)
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal(shape).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.relu() * w).sum().backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch.relu(torch_x) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=RELU_RTOL, atol=RELU_ATOL
    )
    # Exact-zero inputs must have gradient 0 (matches PyTorch).
    zero_mask = x_np == 0.0
    assert np.all(x.grad.numpy()[zero_mask] == 0.0)


def test_relu_grad_fn_set_when_input_requires_grad():
    x_np = _seeded_with_zeros((3, 4))
    x = mojodl.Tensor(x_np, requires_grad=True)

    result = x.relu()

    assert result.grad_fn is not None


def test_relu_under_no_grad_drops_history():
    x_np = _seeded_with_zeros((3, 4))
    x = mojodl.Tensor(x_np, requires_grad=True)

    with mojodl.no_grad():
        result = x.relu()

    assert result.requires_grad is False
    assert result.grad_fn is None


def test_relu_meta_returns_meta_without_allocating():
    mojodl.reset_peak_memory_stats()
    x = mojodl.empty((4, 5), device="meta")
    before = mojodl.memory_stats()

    result = x.relu()

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == (4, 5)
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_relu_meta_backward_fills_meta_grad_without_allocating():
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    x = mojodl.empty((4, 5), device="meta", requires_grad=True)
    w = mojodl.empty((4, 5), device="meta")
    (x.relu() * w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (4, 5)
    assert x.grad.device == "meta"

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_relu_listed_in_all():
    assert "relu" in mojodl.__all__


# ---------------------------------------------------------------------------
# exp
# ---------------------------------------------------------------------------

EXP_RTOL = 1e-6
EXP_ATOL = 1e-7

_EXP_EXTREME_VALUES = np.array(
    [-200.0, -20.0, 0.0, 20.0, 88.0, 89.0, 200.0], dtype=np.float32
)


def _seeded_standard_normal(shape, seed=0):
    """Seeded standard-normal float32 input (no planted specials)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("shape", _FORWARD_SHAPES, ids=["scalar", "1d", "2d", "3d", "empty"])
def test_exp_method_forward_matches_torch(shape):
    x_np = _seeded_standard_normal(shape)
    x = mojodl.Tensor(x_np)

    result = x.exp()
    expected = torch.exp(torch.from_numpy(x_np)).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=EXP_RTOL, atol=EXP_ATOL
    )


@pytest.mark.parametrize("shape", _FORWARD_SHAPES, ids=["scalar", "1d", "2d", "3d", "empty"])
def test_exp_function_forward_matches_torch(shape):
    x_np = _seeded_standard_normal(shape)
    x = mojodl.Tensor(x_np)

    result = mojodl.exp(x)
    expected = torch.exp(torch.from_numpy(x_np)).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=EXP_RTOL, atol=EXP_ATOL
    )


def test_exp_does_not_mutate_input():
    x_np = _seeded_standard_normal((3, 4))
    x = mojodl.Tensor(x_np)

    _ = x.exp()
    _ = mojodl.exp(x)

    np.testing.assert_array_equal(x.numpy(), x_np)


def test_exp_extreme_values_match_torch():
    x = mojodl.Tensor(_EXP_EXTREME_VALUES)
    result = x.exp()
    expected = torch.exp(torch.from_numpy(_EXP_EXTREME_VALUES)).numpy()

    np.testing.assert_allclose(
        result.numpy(), expected, rtol=EXP_RTOL, atol=EXP_ATOL
    )


def test_exp_result_storage_is_accounted():
    shape = (3, 4)
    x_np = _seeded_standard_normal(shape)
    x = mojodl.Tensor(x_np)
    nbytes = int(np.prod(shape)) * 4
    baseline = mojodl.memory_stats()["current_bytes"]

    result = x.exp()

    assert mojodl.memory_stats()["current_bytes"] - baseline == nbytes
    del result
    gc.collect()


@pytest.mark.parametrize(
    "shape",
    [(), (7,), (3, 4), (2, 3, 5)],
    ids=["scalar", "1d", "2d", "3d"],
)
def test_exp_backward_matches_torch(shape):
    x_np = _seeded_standard_normal(shape, seed=0)
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal(shape).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.exp() * w).sum().backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch.exp(torch_x) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=EXP_RTOL, atol=EXP_ATOL
    )


def test_exp_grad_fn_set_when_input_requires_grad():
    x_np = _seeded_standard_normal((3, 4))
    x = mojodl.Tensor(x_np, requires_grad=True)

    result = x.exp()

    assert result.grad_fn is not None


def test_exp_under_no_grad_drops_history():
    x_np = _seeded_standard_normal((3, 4))
    x = mojodl.Tensor(x_np, requires_grad=True)

    with mojodl.no_grad():
        result = x.exp()

    assert result.requires_grad is False
    assert result.grad_fn is None


def test_exp_meta_returns_meta_without_allocating():
    mojodl.reset_peak_memory_stats()
    x = mojodl.empty((4, 5), device="meta")
    before = mojodl.memory_stats()

    result = x.exp()

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == (4, 5)
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_exp_meta_backward_fills_meta_grad_without_allocating():
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    x = mojodl.empty((4, 5), device="meta", requires_grad=True)
    w = mojodl.empty((4, 5), device="meta")
    (x.exp() * w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (4, 5)
    assert x.grad.device == "meta"

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_exp_listed_in_all():
    assert "exp" in mojodl.__all__
