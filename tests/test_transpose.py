"""Tests for Tensor.transpose(dim0, dim1) on cpu and Meta tensors.

PyTorch (CPU) is the numerical oracle. Unlike PyTorch's view, mojodl
transpose returns a new contiguous tensor (kernels require contiguous
row-major storage). Covers forward/backward, consumer contiguity,
memory accounting, Meta dry-run paths, and AxisError / TypeError.
"""

import gc

import numpy as np
import pytest
import torch

import mojodl

_FORWARD_CASES = [
    ((3, 4), 0, 1),
    ((3, 4), 1, 0),
    ((3, 4), -1, -2),
    ((3, 4), 0, 0),
    ((2, 3, 4), 0, 2),
    ((2, 3, 4), 1, 2),
    ((2, 3, 4), 0, -1),
    ((2, 3, 4), -3, 1),
    ((2, 3, 4, 5), 1, 3),
    ((0, 3), 0, 1),
    ((5,), 0, -1),
    ((), 0, 0),
    ((), -1, -1),
]


def _make_data(shape):
    if any(d == 0 for d in shape):
        return np.zeros(shape, dtype=np.float32)
    if shape == ():
        return np.array(1.5, dtype=np.float32)
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize(
    "shape,dim0,dim1",
    _FORWARD_CASES,
    ids=[f"shape{s}_d{d0}_{d1}" for s, d0, d1 in _FORWARD_CASES],
)
def test_transpose_forward_matches_torch(shape, dim0, dim1):
    a_np = _make_data(shape)
    a = mojodl.Tensor(a_np)

    result = a.transpose(dim0, dim1)
    expected = torch.from_numpy(a_np).transpose(dim0, dim1)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == tuple(expected.shape)
    assert result.dtype == mojodl.float32
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_transpose_result_usable_in_matmul():
    rng = np.random.default_rng(0)
    t_np = rng.standard_normal((3, 4)).astype(np.float32)
    m_np = rng.standard_normal((3, 2)).astype(np.float32)

    t = mojodl.Tensor(t_np)
    m = mojodl.Tensor(m_np)
    result = t.transpose(0, 1) @ m

    expected = torch.from_numpy(t_np).transpose(0, 1) @ torch.from_numpy(m_np)

    assert result.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_transpose_result_usable_in_add():
    rng = np.random.default_rng(0)
    t_np = rng.standard_normal((3, 4)).astype(np.float32)
    u_np = rng.standard_normal((4, 3)).astype(np.float32)

    t = mojodl.Tensor(t_np)
    u = mojodl.Tensor(u_np)
    result = t.transpose(0, 1) + u

    expected = torch.from_numpy(t_np).transpose(0, 1) + torch.from_numpy(u_np)

    assert result.shape == tuple(expected.shape)
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_transpose_result_usable_in_reshape():
    rng = np.random.default_rng(0)
    t_np = rng.standard_normal((3, 4)).astype(np.float32)

    t = mojodl.Tensor(t_np)
    result = t.transpose(0, 1).reshape(-1)

    expected = torch.from_numpy(t_np).transpose(0, 1).reshape(-1)

    assert result.shape == tuple(expected.shape)
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_transpose_result_usable_in_sum_dim():
    rng = np.random.default_rng(0)
    t_np = rng.standard_normal((3, 4)).astype(np.float32)

    t = mojodl.Tensor(t_np)
    result = t.transpose(0, 1).sum(dim=0)

    expected = torch.from_numpy(t_np).transpose(0, 1).sum(dim=0)

    assert result.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_transpose_leaves_input_unchanged():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((3, 4)).astype(np.float32)
    a = mojodl.Tensor(a_np)
    before = a.numpy().copy()

    _ = a.transpose(0, 1)

    assert a.shape == (3, 4)
    np.testing.assert_array_equal(a.numpy(), before)


def test_transpose_allocates_new_contiguous_storage():
    t = mojodl.Tensor(np.zeros((64, 32), dtype=np.float32))
    gc.collect()
    mojodl.reset_peak_memory_stats()
    baseline = mojodl.memory_stats()["current_bytes"]

    result = t.transpose(0, 1)

    assert mojodl.memory_stats()["current_bytes"] - baseline == 8192
    assert result.shape == (32, 64)

    del result
    gc.collect()
    assert mojodl.memory_stats()["current_bytes"] == baseline

    del t
    gc.collect()


def test_transpose_backward_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    w_np = rng.standard_normal((4, 3, 2)).astype(np.float32)

    x = mojodl.Tensor(a_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.transpose(0, 2) * w).sum().backward()

    torch_x = torch.tensor(a_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch_x.transpose(0, 2) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (2, 3, 4)
    assert x.grad.dtype == mojodl.float32
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-6, atol=1e-6
    )


def test_transpose_chain_backward_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    w_np = rng.standard_normal((3, 4, 2)).astype(np.float32)

    x = mojodl.Tensor(a_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.transpose(0, 1).transpose(1, 2) * w).sum().backward()

    torch_x = torch.tensor(a_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch_x.transpose(0, 1).transpose(1, 2) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (2, 3, 4)
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-6, atol=1e-6
    )


def test_transpose_matmul_backward_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((3, 4)).astype(np.float32)
    b_np = rng.standard_normal((3, 5)).astype(np.float32)

    a = mojodl.Tensor(a_np, requires_grad=True)
    b = mojodl.Tensor(b_np, requires_grad=True)
    (a.transpose(0, 1) @ b).sum().backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np, requires_grad=True)
    (torch_a.transpose(0, 1) @ torch_b).sum().backward()

    assert isinstance(a.grad, mojodl.Tensor)
    assert isinstance(b.grad, mojodl.Tensor)
    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (3, 5)
    np.testing.assert_allclose(
        a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5
    )


def test_transpose_meta_returns_meta_without_allocating():
    t = mojodl.empty((2, 3, 4), mojodl.float32, device="meta")
    before = mojodl.memory_stats()

    result = t.transpose(0, 2)

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == (4, 3, 2)
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    del t, result
    gc.collect()


def test_transpose_meta_backward_fills_meta_grad_without_allocating():
    before = mojodl.memory_stats()

    x = mojodl.empty((2, 3, 4), device="meta", requires_grad=True)
    x.transpose(-1, 0).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.device == "meta"
    assert x.grad.shape == (2, 3, 4)
    assert x.grad.dtype == mojodl.float32

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


_AXIS_ERROR_CASES = [
    ((2, 3, 4), 0, 3, "3"),
    ((2, 3, 4), -4, 1, "-4"),
    ((), 0, 1, "1"),
]


@pytest.mark.parametrize(
    "shape,dim0,dim1,bad",
    _AXIS_ERROR_CASES,
    ids=["dim3", "dim_neg4", "scalar_dim1"],
)
def test_transpose_out_of_range_raises_axis_error(shape, dim0, dim1, bad):
    a = mojodl.Tensor(np.zeros(shape, dtype=np.float32))

    with pytest.raises(mojodl.AxisError) as exc_info:
        a.transpose(dim0, dim1)

    assert isinstance(exc_info.value, mojodl.ShapeError)
    assert isinstance(exc_info.value, IndexError)
    assert bad in str(exc_info.value)


@pytest.mark.parametrize(
    "shape,dim0,dim1,bad",
    _AXIS_ERROR_CASES,
    ids=["dim3", "dim_neg4", "scalar_dim1"],
)
def test_transpose_out_of_range_on_meta_raises_axis_error(shape, dim0, dim1, bad):
    a = mojodl.empty(shape, mojodl.float32, device="meta")

    with pytest.raises(mojodl.AxisError) as exc_info:
        a.transpose(dim0, dim1)

    assert isinstance(exc_info.value, mojodl.ShapeError)
    assert isinstance(exc_info.value, IndexError)
    assert bad in str(exc_info.value)


@pytest.mark.parametrize(
    "dim0,dim1",
    [(0.0, 1), (True, 1)],
    ids=["float", "bool"],
)
def test_transpose_non_int_raises_type_error(dim0, dim1):
    a = mojodl.Tensor(np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(TypeError):
        a.transpose(dim0, dim1)


def test_transpose_numpy_int_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    t = mojodl.Tensor(a_np)
    torch_t = torch.from_numpy(a_np)

    result = t.transpose(np.int64(0), np.int32(2))
    expected = torch_t.transpose(0, 2)
    assert result.shape == tuple(expected.shape)
    np.testing.assert_array_equal(result.numpy(), expected.numpy())
