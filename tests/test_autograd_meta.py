"""Tests for autograd on Meta tensors via mojodl.empty(..., requires_grad=...).

A Meta tensor carries shape and dtype but no data (GLOSSARY.md). These tests
drive only the public Python API. Dry-run tracing of a training
Step (including .backward()) relies on Meta grads allocating nothing.
"""

import pytest

import mojodl


def test_empty_requires_grad_default_false_on_cpu_and_meta():
    cpu = mojodl.empty((2, 2))
    meta = mojodl.empty((2, 2), device="meta")

    assert cpu.requires_grad is False
    assert cpu.is_leaf is True
    assert cpu.grad is None

    assert meta.requires_grad is False
    assert meta.is_leaf is True
    assert meta.grad is None


def test_empty_requires_grad_true_is_leaf_on_cpu_and_meta():
    cpu = mojodl.empty((2, 2), requires_grad=True)
    x = mojodl.empty((3, 4), device="meta", requires_grad=True)

    assert cpu.requires_grad is True
    assert cpu.is_leaf is True
    assert cpu.grad is None

    assert x.requires_grad is True
    assert x.is_leaf is True
    assert x.grad is None
    assert x.device == "meta"


def test_meta_mul_add_sum_backward_fills_meta_grads():
    x = mojodl.empty((3, 4), device="meta", requires_grad=True)
    w = mojodl.empty((3, 4), device="meta", requires_grad=True)

    loss = ((x * w) + x).sum()

    assert isinstance(loss, mojodl.Tensor)
    assert loss.device == "meta"
    assert loss.shape == ()
    assert loss.grad_fn is not None

    loss.backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.device == "meta"
    assert x.grad.shape == (3, 4)
    assert x.grad.dtype == mojodl.float32
    assert x.grad.requires_grad is False

    assert isinstance(w.grad, mojodl.Tensor)
    assert w.grad.device == "meta"
    assert w.grad.shape == (3, 4)
    assert w.grad.dtype == mojodl.float32
    assert w.grad.requires_grad is False


def test_meta_autograd_allocates_no_bytes():
    before = mojodl.memory_stats()

    x = mojodl.empty((3, 4), device="meta", requires_grad=True)
    w = mojodl.empty((3, 4), device="meta", requires_grad=True)
    loss = ((x * w) + x).sum()
    loss.backward()

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    assert x.grad is not None
    assert w.grad is not None


def test_meta_retain_graph_then_backward_preserves_grad_identity():
    before = mojodl.memory_stats()

    x = mojodl.empty((3, 4), device="meta", requires_grad=True)
    w = mojodl.empty((3, 4), device="meta", requires_grad=True)
    loss = ((x * w) + x).sum()
    loss.backward(retain_graph=True)

    g = x.grad
    assert isinstance(g, mojodl.Tensor)
    assert g.device == "meta"
    assert g.shape == (3, 4)

    loss.backward()

    assert x.grad is g
    assert x.grad.device == "meta"
    assert x.grad.shape == (3, 4)

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


def test_meta_nonscalar_mul_backward_raises_autograd_error():
    x = mojodl.empty((3, 4), device="meta", requires_grad=True)
    w = mojodl.empty((3, 4), device="meta", requires_grad=True)

    assert issubclass(mojodl.AutogradError, RuntimeError)
    with pytest.raises(mojodl.AutogradError) as exc_info:
        (x * w).backward()
    assert isinstance(exc_info.value, RuntimeError)
