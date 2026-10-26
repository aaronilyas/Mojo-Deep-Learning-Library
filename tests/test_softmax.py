"""Tests for Tensor.softmax / mojodl.softmax along a dimension.

PyTorch (CPU) is the numerical oracle. Softmax is float32, so comparisons
use SOFTMAX_RTOL / SOFTMAX_ATOL.
"""

import gc

import numpy as np
import pytest
import torch

import mojodl

SOFTMAX_RTOL = 1e-5
SOFTMAX_ATOL = 1e-6

_FORWARD_CASES = [
    ((), 0),
    ((7,), 0),
    ((7,), -1),
    ((3, 4), 1),
    ((3, 4), 0),
    ((3, 4), -1),
    ((2, 3, 5), 0),
    ((2, 3, 5), 1),
    ((2, 3, 5), 2),
    ((2, 3, 5), -2),
    ((0, 3), 1),
    ((3, 0), 1),
]

_FORWARD_IDS = [
    "scalar_0",
    "1d_0",
    "1d_neg1",
    "2d_1",
    "2d_0",
    "2d_neg1",
    "3d_0",
    "3d_1",
    "3d_2",
    "3d_neg2",
    "empty_rows",
    "empty_cols",
]

_BACKWARD_CASES = [
    ((7,), 0),
    ((3, 4), 1),
    ((3, 4), 0),
    ((2, 3, 5), 1),
    ((2, 3, 5), -1),
]

_BACKWARD_IDS = ["1d_0", "2d_1", "2d_0", "3d_1", "3d_neg1"]

_OOR_CASES = [
    ((3, 4), 2),
    ((3, 4), -3),
    ((7,), 1),
]

_OOR_IDS = ["2d_dim2", "2d_dim_neg3", "1d_dim1"]

_STABILITY_INPUT = np.array(
    [
        [1e4, -1e4, 0.0, 9999.0],
        [-1e4, -9995.0, -1e4, -9999.0],
        [1e4, 1e4, 1e4, 1e4],
    ],
    dtype=np.float32,
)


def _seeded_standard_normal(shape, seed=0):
    """Seeded standard-normal float32 input (no planted specials)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("shape,dim", _FORWARD_CASES, ids=_FORWARD_IDS)
def test_softmax_method_forward_matches_torch(shape, dim):
    x_np = _seeded_standard_normal(shape)
    x = mojodl.Tensor(x_np)

    result = x.softmax(dim)
    expected = torch.softmax(torch.from_numpy(x_np), dim=dim).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=SOFTMAX_RTOL, atol=SOFTMAX_ATOL
    )


@pytest.mark.parametrize("shape,dim", _FORWARD_CASES, ids=_FORWARD_IDS)
def test_softmax_function_forward_matches_torch(shape, dim):
    x_np = _seeded_standard_normal(shape)
    x = mojodl.Tensor(x_np)

    result = mojodl.softmax(x, dim)
    expected = torch.softmax(torch.from_numpy(x_np), dim=dim).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=SOFTMAX_RTOL, atol=SOFTMAX_ATOL
    )


def test_softmax_does_not_mutate_input():
    x_np = _seeded_standard_normal((3, 4))
    x = mojodl.Tensor(x_np)

    _ = x.softmax(1)
    _ = mojodl.softmax(x, 1)

    np.testing.assert_array_equal(x.numpy(), x_np)


@pytest.mark.parametrize("dim", [1, 0], ids=["dim1", "dim0"])
def test_softmax_stability_finite_and_matches_torch(dim):
    x = mojodl.Tensor(_STABILITY_INPUT)
    result = x.softmax(dim)
    expected = torch.softmax(torch.from_numpy(_STABILITY_INPUT), dim=dim).numpy()

    assert np.all(np.isfinite(result.numpy()))
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=SOFTMAX_RTOL, atol=SOFTMAX_ATOL
    )


@pytest.mark.parametrize("dim", [1, 0], ids=["dim1", "dim0"])
def test_softmax_stability_backward_finite_and_matches_torch(dim):
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal(_STABILITY_INPUT.shape).astype(np.float32)

    x = mojodl.Tensor(_STABILITY_INPUT, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.softmax(dim) * w).sum().backward()

    torch_x = torch.tensor(_STABILITY_INPUT, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch.softmax(torch_x, dim=dim) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert np.all(np.isfinite(x.grad.numpy()))
    np.testing.assert_allclose(
        x.grad.numpy(),
        torch_x.grad.numpy(),
        rtol=SOFTMAX_RTOL,
        atol=SOFTMAX_ATOL,
    )


def test_softmax_result_storage_is_accounted():
    shape = (3, 4)
    x_np = _seeded_standard_normal(shape)
    x = mojodl.Tensor(x_np)
    nbytes = int(np.prod(shape)) * 4
    baseline = mojodl.memory_stats()["current_bytes"]

    result = x.softmax(1)

    assert mojodl.memory_stats()["current_bytes"] - baseline == nbytes
    del result
    gc.collect()


@pytest.mark.parametrize("shape,dim", _BACKWARD_CASES, ids=_BACKWARD_IDS)
def test_softmax_backward_matches_torch(shape, dim):
    x_np = _seeded_standard_normal(shape, seed=0)
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal(shape).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.softmax(dim) * w).sum().backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch.softmax(torch_x, dim=dim) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        x.grad.numpy(),
        torch_x.grad.numpy(),
        rtol=SOFTMAX_RTOL,
        atol=SOFTMAX_ATOL,
    )


def test_softmax_grad_fn_set_when_input_requires_grad():
    x_np = _seeded_standard_normal((3, 4))
    x = mojodl.Tensor(x_np, requires_grad=True)

    result = x.softmax(1)

    assert result.grad_fn is not None


def test_softmax_under_no_grad_drops_history():
    x_np = _seeded_standard_normal((3, 4))
    x = mojodl.Tensor(x_np, requires_grad=True)

    with mojodl.no_grad():
        result = x.softmax(1)

    assert result.requires_grad is False
    assert result.grad_fn is None


@pytest.mark.parametrize("shape,dim", _OOR_CASES, ids=_OOR_IDS)
def test_softmax_out_of_range_dim_raises_shape_error(shape, dim):
    x = mojodl.Tensor(np.zeros(shape, dtype=np.float32))

    with pytest.raises(mojodl.ShapeError):
        x.softmax(dim)


@pytest.mark.parametrize("shape,dim", _OOR_CASES, ids=_OOR_IDS)
def test_softmax_meta_out_of_range_dim_raises_shape_error(shape, dim):
    x = mojodl.empty(shape, device="meta")

    with pytest.raises(mojodl.ShapeError):
        x.softmax(dim)


def test_softmax_meta_returns_meta_without_allocating():
    mojodl.reset_peak_memory_stats()
    x = mojodl.empty((4, 5), device="meta")
    before = mojodl.memory_stats()

    result = x.softmax(1)

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == (4, 5)
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_softmax_meta_backward_fills_meta_grad_without_allocating():
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    x = mojodl.empty((4, 5), device="meta", requires_grad=True)
    w = mojodl.empty((4, 5), device="meta")
    (x.softmax(1) * w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (4, 5)
    assert x.grad.device == "meta"

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_softmax_listed_in_all():
    assert "softmax" in mojodl.__all__
