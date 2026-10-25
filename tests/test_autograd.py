"""Tests for requires_grad leaves and .backward() through whole-tensor sum().

PyTorch (CPU) is the numerical oracle. Gradients of + / *, repeated backward,
non-scalar backward, and no_grad are out of scope here.
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
