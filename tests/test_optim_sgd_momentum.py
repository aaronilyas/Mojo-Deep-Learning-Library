"""Acceptance: mojodl.optim.SGD with momentum matches torch.optim.SGD.

Covers multi-step oracle parity, a literal worked example, momentum-buffer
memory accounting, deferred buffer creation when .grad is None, Meta dry-run
with no bytes, and invalid momentum. Semantics match torch.optim.SGD with
dampening=0, nesterov=False, weight_decay=0. PyTorch CPU is the numerical
oracle.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

import mojodl


RTOL = 1e-6
ATOL = 1e-6


# ---------------------------------------------------------------------------
# Oracle match: multi-shape Parameters vs torch.optim.SGD over several steps
# ---------------------------------------------------------------------------


def test_sgd_momentum_step_matches_torch_over_several_steps():
    """Shapes (3, 4), (5,), () with seeded float32 weights; fresh grads each step."""
    rng = np.random.default_rng(0)
    shapes = [(3, 4), (5,), ()]
    weights_np = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    lr = 0.1
    momentum = 0.9
    n_steps = 3

    mojo_params = [
        mojodl.nn.Parameter(mojodl.Tensor(w_np.copy())) for w_np in weights_np
    ]
    torch_params = [
        torch.nn.Parameter(torch.from_numpy(w_np.copy())) for w_np in weights_np
    ]

    mojo_opt = mojodl.optim.SGD(mojo_params, lr=lr, momentum=momentum)
    torch_opt = torch.optim.SGD(torch_params, lr=lr, momentum=momentum)

    for _ in range(n_steps):
        grads_np = [rng.standard_normal(s).astype(np.float32) for s in shapes]
        for mp, tp, g_np in zip(mojo_params, torch_params, grads_np):
            mp.grad = mojodl.Tensor(g_np.copy())
            tp.grad = torch.from_numpy(g_np.copy())

        mojo_opt.step()
        torch_opt.step()

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


def test_sgd_momentum_worked_example_literal():
    """p=[1.0], lr=0.1, momentum=0.9, grad=1.0 → 0.9 after step 1, 0.71 after step 2."""
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0], dtype=np.float32)))
    opt = mojodl.optim.SGD([p], lr=0.1, momentum=0.9)

    p.grad = mojodl.Tensor(np.array([1.0], dtype=np.float32))
    opt.step()
    np.testing.assert_allclose(
        p.numpy(),
        np.array([0.9], dtype=np.float32),
        rtol=RTOL,
        atol=ATOL,
    )

    p.grad = mojodl.Tensor(np.array([1.0], dtype=np.float32))
    opt.step()
    np.testing.assert_allclose(
        p.numpy(),
        np.array([0.71], dtype=np.float32),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# Optimizer state accounting: momentum buffers
# ---------------------------------------------------------------------------


def test_sgd_momentum_buffer_memory_accounting():
    """First step allocates one buffer per param with grad; second step none; del frees."""
    shapes = [(3, 4), (5,), ()]
    nbytes = sum(int(np.prod(s)) * 4 for s in shapes)

    params = [
        mojodl.nn.Parameter(
            mojodl.Tensor(np.zeros(s, dtype=np.float32))
        )
        for s in shapes
    ]
    for p, s in zip(params, shapes):
        p.grad = mojodl.Tensor(np.ones(s, dtype=np.float32))

    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    opt = mojodl.optim.SGD(params, lr=0.1, momentum=0.9)
    opt.step()

    after_first = mojodl.memory_stats()
    assert after_first["current_bytes"] - baseline == nbytes

    mojodl.reset_peak_memory_stats()
    before_second = mojodl.memory_stats()
    opt.step()
    after_second = mojodl.memory_stats()
    assert after_second["current_bytes"] == before_second["current_bytes"]
    assert after_second["peak_bytes"] == before_second["peak_bytes"]

    del opt
    gc.collect()
    assert mojodl.memory_stats()["current_bytes"] == baseline


# ---------------------------------------------------------------------------
# None grad: no buffer on first step; later first update uses buf = grad
# ---------------------------------------------------------------------------


def test_sgd_momentum_skips_none_grad_then_creates_buffer_on_first_update():
    """Param with None grad is unchanged and gets no buffer; later matches torch."""
    with_grad = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([1.0, 2.0], dtype=np.float32))
    )
    with_grad.grad = mojodl.Tensor(np.array([1.0, 1.0], dtype=np.float32))
    deferred = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([10.0, 20.0], dtype=np.float32))
    )
    assert deferred.grad is None
    deferred_before = deferred.numpy().copy()
    nbytes_with_grad = 2 * 4

    torch_with = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    torch_with.grad = torch.tensor([1.0, 1.0])
    torch_def = torch.nn.Parameter(torch.tensor([10.0, 20.0]))
    # torch_def.grad is None

    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    mojo_opt = mojodl.optim.SGD([with_grad, deferred], lr=0.1, momentum=0.9)
    torch_opt = torch.optim.SGD([torch_with, torch_def], lr=0.1, momentum=0.9)

    mojo_opt.step()
    torch_opt.step()

    np.testing.assert_array_equal(deferred.numpy(), deferred_before)
    assert deferred.grad is None
    after_first = mojodl.memory_stats()
    assert after_first["current_bytes"] - baseline == nbytes_with_grad

    np.testing.assert_allclose(
        with_grad.numpy(),
        torch_with.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )

    # Second step: deferred now has a grad; its first update uses buf = grad
    g1 = np.array([0.5, -0.5], dtype=np.float32)
    g2 = np.array([1.0, 1.0], dtype=np.float32)
    with_grad.grad = mojodl.Tensor(g1.copy())
    deferred.grad = mojodl.Tensor(g2.copy())
    torch_with.grad = torch.from_numpy(g1.copy())
    torch_def.grad = torch.from_numpy(g2.copy())

    mojo_opt.step()
    torch_opt.step()

    np.testing.assert_allclose(
        with_grad.numpy(),
        torch_with.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        deferred.numpy(),
        torch_def.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# Meta: no raise, stays meta, no byte allocation across two momentum steps
# ---------------------------------------------------------------------------


def test_sgd_momentum_step_on_meta_parameter_allocates_no_bytes():
    p = mojodl.nn.Parameter(mojodl.empty((3, 4), device="meta"))
    p.grad = mojodl.empty((3, 4), device="meta")

    opt = mojodl.optim.SGD([p], lr=0.1, momentum=0.9)

    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    opt.step()
    opt.step()

    after = mojodl.memory_stats()
    assert p.device == "meta"
    assert p.shape == (3, 4)
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


# ---------------------------------------------------------------------------
# Invalid momentum (PyTorch-like)
# ---------------------------------------------------------------------------


def test_sgd_negative_momentum_raises_value_error():
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0], dtype=np.float32)))
    with pytest.raises(ValueError):
        mojodl.optim.SGD([p], lr=0.1, momentum=-0.5)
