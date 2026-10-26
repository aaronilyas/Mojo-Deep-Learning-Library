"""Tests for mojodl.mse_loss(input, target) with mean reduction.

PyTorch CPU (torch.nn.functional.mse_loss, default reduction "mean") is the
numerical oracle. Covers forward, backward, ShapeError / TypeError, and Meta
dry-run paths. Same-shape only: no silent broadcasting.
"""

import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mojodl

RTOL = 1e-5
ATOL = 1e-6

_FORWARD_SHAPES = [
    (7,),
    (5, 3),
    (2, 3, 4),
]


@pytest.mark.parametrize("shape", _FORWARD_SHAPES, ids=["1d", "2d", "3d"])
def test_mse_loss_forward_matches_torch(shape):
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(shape).astype(np.float32)
    y_np = rng.standard_normal(shape).astype(np.float32)

    x = mojodl.Tensor(x_np)
    y = mojodl.Tensor(y_np)
    result = mojodl.mse_loss(x, y)

    expected = F.mse_loss(torch.from_numpy(x_np), torch.from_numpy(y_np))

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=RTOL, atol=ATOL
    )


def test_mse_loss_identical_input_target_is_exactly_zero():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((5, 3)).astype(np.float32)

    x = mojodl.Tensor(a_np)
    y = mojodl.Tensor(a_np.copy())
    result = mojodl.mse_loss(x, y)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert float(result.numpy()) == 0.0


def test_mse_loss_backward_input_grad_matches_torch():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((5, 3)).astype(np.float32)
    y_np = rng.standard_normal((5, 3)).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np)
    mojodl.mse_loss(x, y).backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np)
    F.mse_loss(torch_x, torch_y).backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (5, 3)
    assert x.grad.dtype == mojodl.float32
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=RTOL, atol=ATOL
    )
    assert y.grad is None


def test_mse_loss_backward_both_grads_match_torch():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((5, 3)).astype(np.float32)
    y_np = rng.standard_normal((5, 3)).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    mojodl.mse_loss(x, y).backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    F.mse_loss(torch_x, torch_y).backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (5, 3)
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=RTOL, atol=ATOL
    )
    assert isinstance(y.grad, mojodl.Tensor)
    assert y.grad.shape == (5, 3)
    np.testing.assert_allclose(
        y.grad.numpy(), torch_y.grad.numpy(), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "input_shape,target_shape",
    [((5, 3), (5, 1)), ((5, 3), (3,))],
    ids=["broadcastable_cols", "rank_mismatch"],
)
def test_mse_loss_shape_mismatch_raises_shape_error(input_shape, target_shape):
    x = mojodl.Tensor(np.zeros(input_shape, dtype=np.float32))
    y = mojodl.Tensor(np.zeros(target_shape, dtype=np.float32))

    assert issubclass(mojodl.ShapeError, ValueError)

    with pytest.raises(mojodl.ShapeError) as exc_info:
        mojodl.mse_loss(x, y)

    msg = str(exc_info.value)
    assert str(input_shape) in msg
    assert str(target_shape) in msg


@pytest.mark.parametrize(
    "bad_arg",
    [
        np.zeros((5, 3), dtype=np.float32),
        1.0,
    ],
    ids=["numpy_array", "float"],
)
def test_mse_loss_non_tensor_input_raises_type_error(bad_arg):
    y = mojodl.Tensor(np.zeros((5, 3), dtype=np.float32))

    with pytest.raises(TypeError):
        mojodl.mse_loss(bad_arg, y)


@pytest.mark.parametrize(
    "bad_arg",
    [
        np.zeros((5, 3), dtype=np.float32),
        1.0,
    ],
    ids=["numpy_array", "float"],
)
def test_mse_loss_non_tensor_target_raises_type_error(bad_arg):
    x = mojodl.Tensor(np.zeros((5, 3), dtype=np.float32))

    with pytest.raises(TypeError):
        mojodl.mse_loss(x, bad_arg)


def test_mse_loss_meta_forward_backward_allocates_no_bytes():
    gc.collect()
    before = mojodl.memory_stats()["current_bytes"]

    x = mojodl.empty((5, 3), device="meta", requires_grad=True)
    y = mojodl.empty((5, 3), device="meta")
    loss = mojodl.mse_loss(x, y)

    assert isinstance(loss, mojodl.Tensor)
    assert loss.device == "meta"
    assert loss.shape == ()
    assert loss.dtype == mojodl.float32

    loss.backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.device == "meta"
    assert x.grad.shape == (5, 3)
    assert x.grad.dtype == mojodl.float32
    assert y.grad is None

    gc.collect()
    after = mojodl.memory_stats()["current_bytes"]
    assert after == before
