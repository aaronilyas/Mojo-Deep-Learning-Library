"""Acceptance: mojodl.optim.SGD without momentum matches torch.optim.SGD.

Covers oracle parity on multi-shape Parameters, a literal worked example,
in-place identity / grad retention / zero allocation, skipping None grads,
generator params from Module.parameters(), Meta dry-run with no bytes, and
invalid-argument errors. PyTorch CPU is the numerical oracle.
Momentum, Adam, and zero_grad are out of scope for this slice.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

import mojodl


RTOL = 1e-6
ATOL = 1e-7


# ---------------------------------------------------------------------------
# Oracle match: multi-shape Parameters vs torch.optim.SGD
# ---------------------------------------------------------------------------


def test_sgd_step_matches_torch_on_seeded_multi_shape_params():
    """Shapes (3, 4), (5,), () with seeded float32 weights and grads."""
    rng = np.random.default_rng(0)
    shapes = [(3, 4), (5,), ()]
    weights_np = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    grads_np = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    lr = 0.1

    mojo_params = []
    for w_np, g_np in zip(weights_np, grads_np):
        p = mojodl.nn.Parameter(mojodl.Tensor(w_np.copy()))
        p.grad = mojodl.Tensor(g_np.copy())
        mojo_params.append(p)

    torch_params = []
    for w_np, g_np in zip(weights_np, grads_np):
        tp = torch.nn.Parameter(torch.from_numpy(w_np.copy()))
        tp.grad = torch.from_numpy(g_np.copy())
        torch_params.append(tp)

    mojodl.optim.SGD(mojo_params, lr=lr).step()
    torch.optim.SGD(torch_params, lr=lr).step()

    for mp, tp in zip(mojo_params, torch_params):
        np.testing.assert_allclose(
            mp.numpy(),
            tp.detach().numpy(),
            rtol=RTOL,
            atol=ATOL,
        )


# ---------------------------------------------------------------------------
# Worked example (independent literal)
# ---------------------------------------------------------------------------


def test_sgd_step_worked_example_literal():
    """p=[1, 2], grad=[0.5, -1], lr=0.1 → p=[0.95, 2.1]."""
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0, 2.0], dtype=np.float32)))
    p.grad = mojodl.Tensor(np.array([0.5, -1.0], dtype=np.float32))

    mojodl.optim.SGD([p], lr=0.1).step()

    np.testing.assert_allclose(
        p.numpy(),
        np.array([0.95, 2.1], dtype=np.float32),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# In place: same object, requires_grad, grad unchanged, no allocation
# ---------------------------------------------------------------------------


def test_sgd_step_updates_in_place_without_allocation():
    p = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    )
    grad = mojodl.Tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    p.grad = grad
    p_id = p
    grad_id = p.grad

    gc.collect()
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    mojodl.optim.SGD([p], lr=0.5).step()

    after = mojodl.memory_stats()

    assert p is p_id
    assert p.requires_grad is True
    assert p.grad is grad_id
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


# ---------------------------------------------------------------------------
# None grad skipped; other param still updated
# ---------------------------------------------------------------------------


def test_sgd_step_skips_param_with_none_grad():
    updated = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([1.0, 2.0], dtype=np.float32))
    )
    updated.grad = mojodl.Tensor(np.array([1.0, 1.0], dtype=np.float32))
    skipped = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([10.0, 20.0], dtype=np.float32))
    )
    assert skipped.grad is None
    skipped_before = skipped.numpy().copy()

    mojodl.optim.SGD([updated, skipped], lr=0.5).step()

    np.testing.assert_allclose(
        updated.numpy(),
        np.array([0.5, 1.5], dtype=np.float32),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_array_equal(skipped.numpy(), skipped_before)
    assert skipped.grad is None


# ---------------------------------------------------------------------------
# Generator from Module.parameters()
# ---------------------------------------------------------------------------


def test_sgd_accepts_parameters_generator_from_linear():
    lin = mojodl.nn.Linear(3, 2, seed=0)
    w_old = lin.weight.numpy().copy()
    b_old = lin.bias.numpy().copy()

    rng = np.random.default_rng(1)
    w_grad = rng.standard_normal(w_old.shape).astype(np.float32)
    b_grad = rng.standard_normal(b_old.shape).astype(np.float32)
    lin.weight.grad = mojodl.Tensor(w_grad)
    lin.bias.grad = mojodl.Tensor(b_grad)

    lr = 0.5
    mojodl.optim.SGD(lin.parameters(), lr=lr).step()

    np.testing.assert_allclose(
        lin.weight.numpy(),
        w_old - lr * w_grad,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        lin.bias.numpy(),
        b_old - lr * b_grad,
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# Meta: no raise, stays meta, no byte allocation
# ---------------------------------------------------------------------------


def test_sgd_step_on_meta_parameter_allocates_no_bytes():
    p = mojodl.nn.Parameter(mojodl.empty((3, 4), device="meta"))
    p.grad = mojodl.empty((3, 4), device="meta")

    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    mojodl.optim.SGD([p], lr=0.1).step()

    after = mojodl.memory_stats()
    assert p.device == "meta"
    assert p.shape == (3, 4)
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


# ---------------------------------------------------------------------------
# Invalid arguments (PyTorch-like)
# ---------------------------------------------------------------------------


def test_sgd_negative_lr_raises_value_error():
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0], dtype=np.float32)))
    with pytest.raises(ValueError):
        mojodl.optim.SGD([p], lr=-0.1)


def test_sgd_empty_param_list_raises_value_error():
    with pytest.raises(ValueError):
        mojodl.optim.SGD([], lr=0.1)


def test_sgd_non_tensor_param_raises_type_error():
    with pytest.raises(TypeError):
        mojodl.optim.SGD([np.array([1.0], dtype=np.float32)], lr=0.1)
