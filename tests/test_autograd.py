"""Tests for requires_grad leaves and .backward() through + / * / sum().

PyTorch (CPU) is the numerical oracle. Repeated backward, non-scalar
backward, no_grad, and broadcasting are out of scope here.
"""

import numpy as np
import pytest
import torch

import mojodl


def test_tensor_default_is_leaf_without_grad():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = mojodl.Tensor(a)

    assert x.requires_grad is False
    assert x.is_leaf is True
    assert x.grad is None
    assert x.grad_fn is None


def test_tensor_requires_grad_true_is_leaf():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = mojodl.Tensor(a, requires_grad=True)

    assert x.requires_grad is True
    assert x.is_leaf is True
    assert x.grad is None
    assert x.grad_fn is None


def test_sum_of_requires_grad_tensor_tracks_grad():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = mojodl.Tensor(a, requires_grad=True)

    s = x.sum()

    assert s.requires_grad is True
    assert s.is_leaf is False
    assert s.grad_fn is not None


def test_sum_without_requires_grad_does_not_track():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    s = mojodl.Tensor(a).sum()

    assert s.requires_grad is False
    assert s.grad_fn is None


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (7,),
        (3, 4),
        (2, 3, 5),
    ],
    ids=["scalar", "1d", "2d", "3d"],
)
def test_sum_backward_fills_leaf_grad_matching_torch(shape):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(shape).astype(np.float32)

    x = mojodl.Tensor(a_np, requires_grad=True)
    x.sum().backward()

    torch_x = torch.tensor(a_np, requires_grad=True)
    torch_x.sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == x.shape
    assert x.grad.dtype == mojodl.float32
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=0, atol=0
    )


def test_grad_after_backward_does_not_require_grad():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = mojodl.Tensor(a, requires_grad=True)

    x.sum().backward()

    assert x.grad.requires_grad is False


def _assert_grad_matches_torch(mojo_leaf, torch_leaf):
    """Compare a mojodl leaf's .grad to the matching torch leaf (or both None)."""
    if torch_leaf.grad is None:
        assert mojo_leaf.grad is None
        return
    assert isinstance(mojo_leaf.grad, mojodl.Tensor)
    assert mojo_leaf.grad.shape == mojo_leaf.shape
    np.testing.assert_allclose(
        mojo_leaf.grad.numpy(),
        torch_leaf.grad.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def _expr_add_sum(x, y):
    return (x + y).sum()


def _expr_mul_sum(x, y):
    return (x * y).sum()


def _expr_square_mul_sum(x, y):
    del y
    return (x * x).sum()


def _expr_double_add_sum(x, y):
    del y
    return (x + x).sum()


def _expr_nested_sum(x, y):
    return ((x * y + y) * x).sum()


def _expr_diamond_sum(x, y):
    a = x * y
    return ((a * a) + a).sum()


@pytest.mark.parametrize(
    "shape,expr",
    [
        ((3, 4), _expr_add_sum),
        ((3, 4), _expr_mul_sum),
        ((3, 4), _expr_square_mul_sum),
        ((3, 4), _expr_double_add_sum),
        ((3, 4), _expr_nested_sum),
        ((3, 4), _expr_diamond_sum),
        ((), _expr_add_sum),
        ((), _expr_mul_sum),
    ],
    ids=[
        "add_sum_2d",
        "mul_sum_2d",
        "square_mul_sum_2d",
        "double_add_sum_2d",
        "nested_sum_2d",
        "diamond_sum_2d",
        "add_sum_scalar",
        "mul_sum_scalar",
    ],
)
def test_add_mul_sum_backward_matches_torch(shape, expr):
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(shape).astype(np.float32)
    y_np = rng.standard_normal(shape).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    expr(x, y).backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    expr(torch_x, torch_y).backward()

    _assert_grad_matches_torch(x, torch_x)
    _assert_grad_matches_torch(y, torch_y)


def test_non_requiring_grad_operand_stays_none_grad():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((3, 4)).astype(np.float32)
    w_np = rng.standard_normal((3, 4)).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    w = mojodl.Tensor(w_np, requires_grad=False)
    ((x * w) + w).sum().backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_w = torch.tensor(w_np, requires_grad=False)
    ((torch_x * torch_w) + torch_w).sum().backward()

    _assert_grad_matches_torch(x, torch_x)
    assert w.grad is None
    assert torch_w.grad is None


def test_add_mul_intermediates_track_grad_when_inputs_require():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((3, 4)).astype(np.float32)
    y_np = rng.standard_normal((3, 4)).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)

    prod = x * y
    assert prod.requires_grad is True
    assert prod.is_leaf is False
    assert prod.grad_fn is not None

    total = prod + y
    assert total.requires_grad is True
    assert total.is_leaf is False
    assert total.grad_fn is not None


def test_add_mul_without_requires_grad_do_not_track():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    x = mojodl.Tensor(a)
    y = mojodl.Tensor(b)

    added = x + y
    assert added.requires_grad is False
    assert added.grad_fn is None

    multiplied = x * y
    assert multiplied.requires_grad is False
    assert multiplied.grad_fn is None


def test_leaf_grads_are_distinct_objects_after_add_backward():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((3, 4)).astype(np.float32)
    y_np = rng.standard_normal((3, 4)).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    (x + y).sum().backward()

    assert x.grad is not None
    assert y.grad is not None
    assert x.grad is not y.grad
