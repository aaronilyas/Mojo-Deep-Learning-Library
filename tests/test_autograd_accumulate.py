"""Tests for gradient accumulation across .backward() calls.

PyTorch (CPU) is the numerical oracle. Covers accumulation, zeroing,
retain_graph, AutogradError on freed saved tensors, and release of saved
tensors after backward (memory counters are process-wide: assert deltas
from a baseline; gc.collect() only before taking the baseline).
"""

import gc

import numpy as np
import pytest
import torch

import mojodl


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


def _random_xy(shape=(3, 4), seed=0):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(shape).astype(np.float32)
    y_np = rng.standard_normal(shape).astype(np.float32)
    return x_np, y_np


def test_recompute_twice_accumulates_matching_torch():
    """Fresh graph each time: two backwards double the leaf grads."""
    x_np, y_np = _random_xy()

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    ((x * y) + x).sum().backward()
    ((x * y) + x).sum().backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    ((torch_x * torch_y) + torch_x).sum().backward()
    ((torch_x * torch_y) + torch_x).sum().backward()

    _assert_grad_matches_torch(x, torch_x)
    _assert_grad_matches_torch(y, torch_y)


def test_same_graph_twice_with_retain_graph_matches_torch():
    """retain_graph=True then a second backward on the same loss."""
    x_np, y_np = _random_xy()

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    loss = ((x * y) + x).sum()
    loss.backward(retain_graph=True)
    loss.backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    torch_loss = ((torch_x * torch_y) + torch_x).sum()
    torch_loss.backward(retain_graph=True)
    torch_loss.backward()

    _assert_grad_matches_torch(x, torch_x)
    _assert_grad_matches_torch(y, torch_y)


def test_add_only_graph_backward_twice_matches_torch():
    """Graphs that save no tensors can be traversed twice without retain_graph."""
    x_np, y_np = _random_xy()

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    s = (x + y).sum()
    s.backward()
    s.backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    torch_s = (torch_x + torch_y).sum()
    torch_s.backward()
    torch_s.backward()

    _assert_grad_matches_torch(x, torch_x)
    _assert_grad_matches_torch(y, torch_y)


def test_grad_identity_preserved_across_accumulation():
    """x.grad object identity is stable; values accumulate in place."""
    x_np, y_np = _random_xy()

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    ((x * y) + x).sum().backward()
    g = x.grad
    assert g is not None

    ((x * y) + x).sum().backward()

    assert x.grad is g
    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    ((torch_x * torch_y) + torch_x).sum().backward()
    ((torch_x * torch_y) + torch_x).sum().backward()
    np.testing.assert_allclose(
        g.numpy(), torch_x.grad.numpy(), rtol=1e-6, atol=1e-6
    )


def test_zeroing_grad_then_fresh_backward_matches_single_torch():
    """Assigning .grad = None clears accumulation for the next backward."""
    x_np, y_np = _random_xy()

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    ((x * y) + x).sum().backward()
    x.grad = None
    y.grad = None
    ((x * y) + x).sum().backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    ((torch_x * torch_y) + torch_x).sum().backward()

    _assert_grad_matches_torch(x, torch_x)
    _assert_grad_matches_torch(y, torch_y)


def test_second_backward_without_retain_graph_raises_autograd_error():
    """Mul saves tensors; freed after backward → AutogradError on reuse."""
    x_np, y_np = _random_xy()

    x = mojodl.Tensor(x_np, requires_grad=True)
    y = mojodl.Tensor(y_np, requires_grad=True)
    loss = (x * y).sum()
    loss.backward()

    assert issubclass(mojodl.AutogradError, RuntimeError)
    with pytest.raises(mojodl.AutogradError, match="retain_graph") as exc_info:
        loss.backward()
    assert isinstance(exc_info.value, RuntimeError)


@pytest.mark.parametrize(
    "retain_graph,expected_after_backward",
    [
        (False, lambda N: 4 * N + 4),
        (True, lambda N: 5 * N + 4),
    ],
    ids=["release_saved", "retain_saved"],
)
def test_backward_releases_or_retains_saved_tensors(retain_graph, expected_after_backward):
    """Saved mul intermediate is freed unless retain_graph=True.

    Layout (N = 64*64*4): after building `loss = (a * a).sum()` and `del a`,
    only x, y, saved `a`, and the scalar loss remain (3*N + 4). After
    backward(False): a gone, x.grad and y.grad added (4*N + 4). After
    backward(True): a kept (5*N + 4). Refcounting alone must free — no
    gc.collect() between steps.
    """
    shape = (64, 64)
    N = 64 * 64 * 4  # 16384

    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    rng = np.random.default_rng(0)
    x = mojodl.Tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=True)
    y = mojodl.Tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=True)
    a = x * y
    loss = (a * a).sum()
    del a

    assert mojodl.memory_stats()["current_bytes"] - baseline == 3 * N + 4

    if retain_graph:
        loss.backward(retain_graph=True)
    else:
        loss.backward()

    assert (
        mojodl.memory_stats()["current_bytes"] - baseline
        == expected_after_backward(N)
    )
