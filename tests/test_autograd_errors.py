"""Tests for AutogradError on non-scalar / no-grad .backward() and .grad assign.

PyTorch (CPU) is the oracle: implicit backward needs a one-element output
(shape (), (1,), (1, 1) all work); otherwise RuntimeError mentioning
"scalar". A tensor that does not require grad raises RuntimeError mentioning
"does not require grad". Assigning .grad with a mismatched shape or device
raises RuntimeError (mojodl: AutogradError); None or a matching Tensor is
allowed; a non-Tensor raises TypeError. mojodl maps the RuntimeError cases
to AutogradError (a RuntimeError subclass).
"""

import numpy as np
import pytest
import torch

import mojodl


def test_nonscalar_mul_backward_raises_autograd_error():
    """(x * x).backward() on shape (3, 4) must raise; leaf grad stays None."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    x = mojodl.Tensor(a, requires_grad=True)

    assert issubclass(mojodl.AutogradError, RuntimeError)
    with pytest.raises(mojodl.AutogradError, match="scalar") as exc_info:
        (x * x).backward()
    assert isinstance(exc_info.value, RuntimeError)
    assert x.grad is None


def test_nonscalar_leaf_backward_raises_autograd_error():
    """Leaf .backward() on shape (2,) must raise; leaf grad stays None."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2,)).astype(np.float32)
    x = mojodl.Tensor(a, requires_grad=True)

    assert issubclass(mojodl.AutogradError, RuntimeError)
    with pytest.raises(mojodl.AutogradError, match="scalar") as exc_info:
        x.backward()
    assert isinstance(exc_info.value, RuntimeError)
    assert x.grad is None


@pytest.mark.parametrize(
    "shape",
    [(1,), (1, 1)],
    ids=["shape_1", "shape_1x1"],
)
def test_one_element_mul_backward_matches_torch(shape):
    """One-element non-0-d outputs seed ones and fill leaf grad like torch."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape).astype(np.float32)

    x = mojodl.Tensor(a, requires_grad=True)
    (x * x).backward()

    torch_x = torch.tensor(a, requires_grad=True)
    (torch_x * torch_x).backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == x.shape
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-6, atol=1e-6
    )


def test_no_grad_sum_backward_raises_autograd_error():
    """sum().backward() when nothing requires grad must raise AutogradError."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    assert issubclass(mojodl.AutogradError, RuntimeError)
    with pytest.raises(mojodl.AutogradError, match="does not require grad") as exc_info:
        mojodl.Tensor(a).sum().backward()
    assert isinstance(exc_info.value, RuntimeError)


def test_grad_assign_shape_mismatch_raises_autograd_error():
    """x.grad = wrong-shape Tensor must raise; leaf grad stays None."""
    x = mojodl.Tensor([1.0, 2.0], requires_grad=True)

    assert issubclass(mojodl.AutogradError, RuntimeError)
    with pytest.raises(mojodl.AutogradError) as exc_info:
        x.grad = mojodl.Tensor([9.0])
    msg = str(exc_info.value)
    assert "(1,)" in msg
    assert "(2,)" in msg
    assert isinstance(exc_info.value, RuntimeError)
    assert x.grad is None


def test_grad_assign_device_mismatch_raises_autograd_error():
    """x.grad = meta Tensor onto cpu leaf must raise AutogradError."""
    x = mojodl.Tensor([1.0, 2.0], requires_grad=True)

    assert issubclass(mojodl.AutogradError, RuntimeError)
    with pytest.raises(mojodl.AutogradError) as exc_info:
        x.grad = mojodl.empty((2,), device="meta")
    msg = str(exc_info.value)
    assert "meta" in msg
    assert "cpu" in msg
    assert isinstance(exc_info.value, RuntimeError)


def test_grad_assign_matching_then_backward_accumulates():
    """Same-shape same-device .grad assign then backward accumulates onto it."""
    x = mojodl.Tensor([1.0, 2.0], requires_grad=True)
    x.grad = mojodl.Tensor([10.0, 20.0])
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.array([12.0, 24.0], dtype=np.float32))


def test_grad_assign_none_allowed_non_tensor_raises_type_error():
    """x.grad = None clears; x.grad = float raises TypeError."""
    x = mojodl.Tensor([1.0, 2.0], requires_grad=True)
    x.grad = mojodl.Tensor([10.0, 20.0])
    x.grad = None
    assert x.grad is None
    with pytest.raises(TypeError):
        x.grad = 3.0
    assert x.grad is None
