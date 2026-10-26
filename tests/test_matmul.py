"""Tests for 2-D Tensor @ Tensor matmul forward, backward, errors, and Meta.

PyTorch (CPU) is the numerical oracle. Batched/broadcast matmul and 1-D
matmul semantics are out of scope beyond ShapeError cases.
"""

import gc

import numpy as np
import pytest
import torch

import mojodl

MATMUL_RTOL = 1e-5
MATMUL_ATOL = 1e-5

_NON_SQUARE_SHAPES = [
    ((2, 3), (3, 4)),
    ((5, 1), (1, 7)),
    ((1, 6), (6, 1)),
    ((4, 8), (8, 3)),
]


def _seeded_pair(a_shape, b_shape, seed=0):
    rng = np.random.default_rng(seed)
    a_np = rng.standard_normal(a_shape).astype(np.float32)
    b_np = rng.standard_normal(b_shape).astype(np.float32)
    return a_np, b_np


@pytest.mark.parametrize("a_shape,b_shape", _NON_SQUARE_SHAPES)
def test_matmul_forward_non_square_matches_torch(a_shape, b_shape):
    a_np, b_np = _seeded_pair(a_shape, b_shape)
    a = mojodl.Tensor(a_np)
    b = mojodl.Tensor(b_np)

    result = a @ b
    expected = (torch.from_numpy(a_np) @ torch.from_numpy(b_np)).numpy()

    m, n = a_shape[0], b_shape[1]
    assert isinstance(result, mojodl.Tensor)
    assert result.shape == (m, n)
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected, rtol=MATMUL_RTOL, atol=MATMUL_ATOL
    )


def test_matmul_zero_size_inner_dim_gives_zeros():
    a = mojodl.Tensor(np.zeros((3, 0), dtype=np.float32))
    b = mojodl.Tensor(np.zeros((0, 2), dtype=np.float32))

    result = a @ b

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == (3, 2)
    assert result.dtype == mojodl.float32
    np.testing.assert_array_equal(result.numpy(), np.zeros((3, 2), dtype=np.float32))


def test_matmul_zero_size_outer_dim_gives_empty_rows():
    a = mojodl.Tensor(np.zeros((0, 4), dtype=np.float32))
    b = mojodl.Tensor(np.zeros((4, 2), dtype=np.float32))

    result = a @ b

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == (0, 2)
    assert result.dtype == mojodl.float32


def test_matmul_does_not_mutate_inputs():
    a_np, b_np = _seeded_pair((2, 3), (3, 4))
    a = mojodl.Tensor(a_np)
    b = mojodl.Tensor(b_np)

    _ = a @ b

    np.testing.assert_array_equal(a.numpy(), a_np)
    np.testing.assert_array_equal(b.numpy(), b_np)


def test_matmul_result_storage_is_accounted():
    a_shape, b_shape = (4, 8), (8, 3)
    a_np, b_np = _seeded_pair(a_shape, b_shape)
    a = mojodl.Tensor(a_np)
    b = mojodl.Tensor(b_np)
    m, n = a_shape[0], b_shape[1]
    nbytes = m * n * 4
    baseline = mojodl.memory_stats()["current_bytes"]

    result = a @ b

    assert mojodl.memory_stats()["current_bytes"] - baseline == nbytes
    del result
    gc.collect()


@pytest.mark.parametrize("a_shape,b_shape", _NON_SQUARE_SHAPES)
def test_matmul_backward_both_require_grad_matches_torch(a_shape, b_shape):
    a_np, b_np = _seeded_pair(a_shape, b_shape, seed=0)
    m, n = a_shape[0], b_shape[1]
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal((m, n)).astype(np.float32)

    a = mojodl.Tensor(a_np, requires_grad=True)
    b = mojodl.Tensor(b_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    ((a @ b) * w).sum().backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    ((torch_a @ torch_b) * torch_w).sum().backward()

    assert isinstance(a.grad, mojodl.Tensor)
    assert isinstance(b.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        a.grad.numpy(), torch_a.grad.numpy(), rtol=MATMUL_RTOL, atol=MATMUL_ATOL
    )
    np.testing.assert_allclose(
        b.grad.numpy(), torch_b.grad.numpy(), rtol=MATMUL_RTOL, atol=MATMUL_ATOL
    )


@pytest.mark.parametrize("a_shape,b_shape", _NON_SQUARE_SHAPES)
def test_matmul_backward_only_a_requires_grad(a_shape, b_shape):
    a_np, b_np = _seeded_pair(a_shape, b_shape, seed=0)
    m, n = a_shape[0], b_shape[1]
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal((m, n)).astype(np.float32)

    a = mojodl.Tensor(a_np, requires_grad=True)
    b = mojodl.Tensor(b_np)
    w = mojodl.Tensor(w_np)
    ((a @ b) * w).sum().backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np)
    torch_w = torch.tensor(w_np)
    ((torch_a @ torch_b) * torch_w).sum().backward()

    assert b.grad is None
    assert isinstance(a.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        a.grad.numpy(), torch_a.grad.numpy(), rtol=MATMUL_RTOL, atol=MATMUL_ATOL
    )


@pytest.mark.parametrize("a_shape,b_shape", _NON_SQUARE_SHAPES)
def test_matmul_backward_only_b_requires_grad(a_shape, b_shape):
    a_np, b_np = _seeded_pair(a_shape, b_shape, seed=0)
    m, n = a_shape[0], b_shape[1]
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal((m, n)).astype(np.float32)

    a = mojodl.Tensor(a_np)
    b = mojodl.Tensor(b_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    ((a @ b) * w).sum().backward()

    torch_a = torch.tensor(a_np)
    torch_b = torch.tensor(b_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    ((torch_a @ torch_b) * torch_w).sum().backward()

    assert a.grad is None
    assert isinstance(b.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        b.grad.numpy(), torch_b.grad.numpy(), rtol=MATMUL_RTOL, atol=MATMUL_ATOL
    )


def test_matmul_grad_fn_set_when_input_requires_grad():
    a_np, b_np = _seeded_pair((2, 3), (3, 4))
    a = mojodl.Tensor(a_np, requires_grad=True)
    b = mojodl.Tensor(b_np)

    result = a @ b

    assert result.grad_fn is not None


def test_matmul_under_no_grad_drops_history():
    a_np, b_np = _seeded_pair((2, 3), (3, 4))
    a = mojodl.Tensor(a_np, requires_grad=True)
    b = mojodl.Tensor(b_np, requires_grad=True)

    with mojodl.no_grad():
        result = a @ b

    assert result.requires_grad is False
    assert result.grad_fn is None


def test_matmul_mismatched_inner_dims_raises_shape_error():
    a = mojodl.Tensor(np.zeros((2, 3), dtype=np.float32))
    b = mojodl.Tensor(np.zeros((4, 5), dtype=np.float32))

    with pytest.raises(mojodl.ShapeError) as exc_info:
        a @ b

    msg = str(exc_info.value)
    assert "(2, 3)" in msg
    assert "(4, 5)" in msg


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((3,), (3, 4)),
        ((2, 3), (3,)),
        ((), ()),
        ((2, 2, 2), (2, 2)),
    ],
    ids=["1d_left", "1d_right", "scalars", "3d_left"],
)
def test_matmul_non_2d_raises_shape_error(a_shape, b_shape):
    a = mojodl.Tensor(np.zeros(a_shape, dtype=np.float32))
    b = mojodl.Tensor(np.zeros(b_shape, dtype=np.float32))

    with pytest.raises(mojodl.ShapeError):
        a @ b


def test_matmul_cpu_and_meta_raises_device_error():
    a = mojodl.Tensor(np.zeros((2, 3), dtype=np.float32))
    b = mojodl.empty((3, 4), device="meta")

    with pytest.raises(mojodl.DeviceError):
        a @ b

    del a, b
    gc.collect()


def test_matmul_scalar_raises_type_error():
    a = mojodl.Tensor(np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(TypeError):
        a @ 2.0


def test_matmul_meta_compatible_returns_meta_without_allocating():
    mojodl.reset_peak_memory_stats()
    a = mojodl.empty((5, 3), device="meta")
    b = mojodl.empty((3, 7), device="meta")
    before = mojodl.memory_stats()

    result = a @ b

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == (5, 7)
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_matmul_meta_mismatched_inner_dims_raises_shape_error():
    a = mojodl.empty((2, 3), device="meta")
    b = mojodl.empty((4, 5), device="meta")

    with pytest.raises(mojodl.ShapeError):
        a @ b


def test_matmul_meta_backward_fills_meta_grads_without_allocating():
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    a = mojodl.empty((5, 3), device="meta", requires_grad=True)
    b = mojodl.empty((3, 7), device="meta", requires_grad=True)
    w = mojodl.empty((5, 7), device="meta")
    ((a @ b) * w).sum().backward()

    assert isinstance(a.grad, mojodl.Tensor)
    assert a.grad.shape == (5, 3)
    assert a.grad.device == "meta"
    assert isinstance(b.grad, mojodl.Tensor)
    assert b.grad.shape == (3, 7)
    assert b.grad.device == "meta"

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
