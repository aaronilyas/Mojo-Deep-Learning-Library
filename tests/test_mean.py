"""Tests for Tensor.mean(dim=None, keepdim=False) on cpu and Meta tensors.

PyTorch (CPU) is the numerical oracle. Covers one-axis and whole-tensor
reduction forward/backward, Meta dry-run paths, and AxisError on bad dims.
"""

import gc
import itertools

import numpy as np
import pytest
import torch

import mojodl


def _forward_cases():
    shapes = [(5,), (3, 4), (2, 3, 4)]
    for shape in shapes:
        yield shape, None, False
        ndim = len(shape)
        for dim, keepdim in itertools.product(range(-ndim, ndim), (False, True)):
            yield shape, dim, keepdim


@pytest.mark.parametrize(
    "shape,dim,keepdim",
    list(_forward_cases()),
    ids=[f"shape{s}_dim{d}_keepdim{k}" for s, d, k in _forward_cases()],
)
def test_mean_dim_matches_torch(shape, dim, keepdim):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(shape).astype(np.float32)
    a = mojodl.Tensor(a_np)

    if dim is None:
        result = a.mean()
        expected = torch.from_numpy(a_np).mean()
    else:
        result = a.mean(dim=dim, keepdim=keepdim)
        expected = torch.from_numpy(a_np).mean(dim=dim, keepdim=keepdim)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == tuple(expected.shape)
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_mean_dim_positional_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    a = mojodl.Tensor(a_np)

    result = a.mean(1)
    expected = torch.from_numpy(a_np).mean(1)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == tuple(expected.shape)
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_mean_no_arg_on_scalar_matches_torch():
    a_np = np.array(3.5, dtype=np.float32)
    a = mojodl.Tensor(a_np)

    result = a.mean()
    expected = torch.from_numpy(a_np).mean()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dim", [0, -1], ids=["dim0", "dim_neg1"])
def test_mean_dim_on_scalar_matches_torch(dim):
    a_np = np.array(3.5, dtype=np.float32)
    a = mojodl.Tensor(a_np)

    result = a.mean(dim=dim)
    expected = torch.from_numpy(a_np).mean(dim=dim)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_mean_dim_none_keepdim_true_is_ones_shape():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    a = mojodl.Tensor(a_np)

    result = a.mean(dim=None, keepdim=True)
    ref = a_np.astype(np.float64).mean()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == (1, 1, 1)
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(result.numpy(), np.full((1, 1, 1), ref), rtol=1e-5)


def test_mean_empty_1d_is_nan():
    a = mojodl.Tensor(np.zeros((0,), dtype=np.float32))

    result = a.mean()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    assert np.isnan(result.numpy())


def test_mean_dim_empty_along_zero_length_axis():
    a = mojodl.Tensor(np.zeros((2, 0), dtype=np.float32))

    result = a.mean(dim=1)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == (2,)
    assert result.dtype == mojodl.float32
    assert np.all(np.isnan(result.numpy()))


def test_mean_dim_empty_collapsing_nonzero_axis():
    a = mojodl.Tensor(np.zeros((2, 0), dtype=np.float32))

    result = a.mean(dim=0)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == (0,)
    assert result.dtype == mojodl.float32


def test_mean_wide_accumulation_matches_float64():
    rng = np.random.default_rng(0)
    a_np = rng.random((4, 2**20), dtype=np.float32)
    a = mojodl.Tensor(a_np)

    result = a.mean()
    ref = a_np.astype(np.float64).mean()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    np.testing.assert_allclose(float(result.numpy()), ref, rtol=1e-6)

    del a, result, a_np
    gc.collect()


def test_mean_dim_wide_accumulation_matches_float64():
    rng = np.random.default_rng(0)
    a_np = rng.random((4, 2**20), dtype=np.float32)
    a = mojodl.Tensor(a_np)

    result = a.mean(dim=1)
    ref = a_np.astype(np.float64).mean(axis=1)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == (4,)
    np.testing.assert_allclose(result.numpy().astype(np.float64), ref, rtol=1e-6)

    del a, result, a_np
    gc.collect()


@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
def test_mean_dim_result_storage_is_accounted(keepdim):
    a = mojodl.Tensor(np.zeros((3, 4), dtype=np.float32))
    baseline = mojodl.memory_stats()["current_bytes"]

    result = a.mean(dim=1, keepdim=keepdim)

    assert mojodl.memory_stats()["current_bytes"] - baseline == 12
    del result
    gc.collect()


def test_mean_result_storage_is_accounted():
    a = mojodl.Tensor(np.zeros((3, 4), dtype=np.float32))
    baseline = mojodl.memory_stats()["current_bytes"]

    result = a.mean()

    assert mojodl.memory_stats()["current_bytes"] - baseline == 4
    del result
    gc.collect()


def _backward_cases():
    for shape in [(3, 4), (2, 3, 4)]:
        yield shape, None, False
        yield shape, None, True
        ndim = len(shape)
        for dim, keepdim in itertools.product(range(-ndim, ndim), (False, True)):
            yield shape, dim, keepdim


@pytest.mark.parametrize(
    "shape,dim,keepdim",
    list(_backward_cases()),
    ids=[f"shape{s}_dim{d}_keepdim{k}" for s, d, k in _backward_cases()],
)
def test_mean_dim_backward_matches_torch(shape, dim, keepdim):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(shape).astype(np.float32)
    if dim is None:
        out_shape = tuple(torch.from_numpy(a_np).mean(dim=None, keepdim=keepdim).shape)
    else:
        out_shape = tuple(
            torch.from_numpy(a_np).mean(dim=dim, keepdim=keepdim).shape
        )
    w_np = rng.standard_normal(out_shape).astype(np.float32)

    x = mojodl.Tensor(a_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    if dim is None:
        (x.mean(dim=None, keepdim=keepdim) * w).sum().backward()
    else:
        (x.mean(dim=dim, keepdim=keepdim) * w).sum().backward()

    torch_x = torch.tensor(a_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    if dim is None:
        (torch_x.mean(dim=None, keepdim=keepdim) * torch_w).sum().backward()
    else:
        (torch_x.mean(dim=dim, keepdim=keepdim) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == x.shape
    assert x.grad.dtype == mojodl.float32
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    "keepdim,expected_shape",
    [(False, (2, 4)), (True, (2, 1, 4))],
    ids=["keepdim_false", "keepdim_true"],
)
def test_mean_dim_meta_returns_meta_without_allocating(keepdim, expected_shape):
    t = mojodl.empty((2, 3, 4), mojodl.float32, device="meta")
    before = mojodl.memory_stats()

    result = t.mean(dim=1, keepdim=keepdim)

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == expected_shape
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    del t, result
    gc.collect()


def test_mean_meta_returns_meta_scalar_without_allocating():
    t = mojodl.empty((2, 3, 4), mojodl.float32, device="meta")
    before = mojodl.memory_stats()

    result = t.mean()

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    del t, result
    gc.collect()


def test_mean_dim_meta_backward_fills_meta_grad_without_allocating():
    before = mojodl.memory_stats()

    x = mojodl.empty((2, 3, 4), device="meta", requires_grad=True)
    x.mean(dim=0).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.device == "meta"
    assert x.grad.shape == (2, 3, 4)
    assert x.grad.dtype == mojodl.float32

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_mean_meta_backward_fills_meta_grad_without_allocating():
    before = mojodl.memory_stats()

    x = mojodl.empty((2, 3, 4), device="meta", requires_grad=True)
    x.mean().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.device == "meta"
    assert x.grad.shape == (2, 3, 4)
    assert x.grad.dtype == mojodl.float32

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((2, 3, 4), 3),
        ((2, 3, 4), -4),
        ((), 1),
    ],
    ids=["dim3", "dim_neg4", "scalar_dim1"],
)
def test_mean_dim_out_of_range_raises_axis_error(shape, dim):
    a = mojodl.Tensor(np.zeros(shape, dtype=np.float32))

    with pytest.raises(mojodl.AxisError) as exc_info:
        a.mean(dim=dim)

    assert isinstance(exc_info.value, mojodl.ShapeError)
    assert isinstance(exc_info.value, IndexError)
    assert str(dim) in str(exc_info.value)


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((2, 3, 4), 3),
        ((2, 3, 4), -4),
        ((), 1),
    ],
    ids=["dim3", "dim_neg4", "scalar_dim1"],
)
def test_mean_dim_out_of_range_on_meta_raises_axis_error(shape, dim):
    a = mojodl.empty(shape, mojodl.float32, device="meta")

    with pytest.raises(mojodl.AxisError) as exc_info:
        a.mean(dim=dim)

    assert isinstance(exc_info.value, mojodl.ShapeError)
    assert isinstance(exc_info.value, IndexError)
    assert str(dim) in str(exc_info.value)


@pytest.mark.parametrize("dim", [1.0, True], ids=["float", "bool"])
def test_mean_dim_non_int_raises_type_error(dim):
    a = mojodl.Tensor(np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(TypeError):
        a.mean(dim=dim)
