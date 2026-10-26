"""Acceptance: mojodl.optim.Adam matches torch.optim.Adam.

Covers multi-step oracle parity (defaults and custom hyperparameters), a
literal worked example, fp32 moment-state memory accounting, deferred state
when .grad is None, Meta dry-run with no bytes, and invalid arguments.
Semantics match torch.optim.Adam with weight_decay=0, amsgrad=False. PyTorch
CPU is the numerical oracle.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

import mojodl


RTOL = 1e-5
ATOL = 1e-6


# ---------------------------------------------------------------------------
# Oracle match: multi-shape Parameters vs torch.optim.Adam over several steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lr,betas,eps",
    [
        (1e-3, (0.9, 0.999), 1e-8),
        (0.01, (0.8, 0.99), 1e-6),
    ],
    ids=["defaults", "custom"],
)
def test_adam_step_matches_torch_over_several_steps(lr, betas, eps):
    """Shapes (3, 4), (5,), () with seeded float32 weights; fresh grads each step."""
    rng = np.random.default_rng(0)
    shapes = [(3, 4), (5,), ()]
    weights_np = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    n_steps = 3

    mojo_params = [
        mojodl.nn.Parameter(mojodl.Tensor(w_np.copy())) for w_np in weights_np
    ]
    torch_params = [
        torch.nn.Parameter(torch.from_numpy(w_np.copy())) for w_np in weights_np
    ]

    mojo_opt = mojodl.optim.Adam(mojo_params, lr=lr, betas=betas, eps=eps)
    torch_opt = torch.optim.Adam(torch_params, lr=lr, betas=betas, eps=eps)

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


def test_adam_worked_example_literal():
    """p=[1.0, -2.0], grad=[0.5, -3.0], lr=0.1 → ≈[0.9, -1.9] (first step ≈ lr·sign)."""
    p = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([1.0, -2.0], dtype=np.float32))
    )
    p.grad = mojodl.Tensor(np.array([0.5, -3.0], dtype=np.float32))

    mojodl.optim.Adam([p], lr=0.1).step()

    np.testing.assert_allclose(
        p.numpy(),
        np.array([0.9, -1.9], dtype=np.float32),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# Optimizer state accounting: two fp32 moment buffers per parameter
# ---------------------------------------------------------------------------


def test_adam_moment_state_memory_accounting():
    """First step allocates 2× param bytes; second step none; del frees."""
    shapes = [(3, 4), (5,), ()]
    nbytes = sum(int(np.prod(s)) * 4 for s in shapes)

    params = [
        mojodl.nn.Parameter(mojodl.Tensor(np.zeros(s, dtype=np.float32)))
        for s in shapes
    ]
    for p, s in zip(params, shapes):
        p.grad = mojodl.Tensor(np.ones(s, dtype=np.float32))

    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    opt = mojodl.optim.Adam(params)
    opt.step()

    after_first = mojodl.memory_stats()
    assert after_first["current_bytes"] - baseline == 2 * nbytes

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
# None grad: no state on first step; later first update starts step count at 1
# ---------------------------------------------------------------------------


def test_adam_skips_none_grad_then_creates_state_on_first_update():
    """Param with None grad is unchanged and gets no state; later matches torch."""
    with_grad = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([1.0, 2.0], dtype=np.float32))
    )
    with_grad.grad = mojodl.Tensor(np.array([1.0, 1.0], dtype=np.float32))
    deferred = mojodl.nn.Parameter(
        mojodl.Tensor(np.array([10.0, 20.0], dtype=np.float32))
    )
    assert deferred.grad is None
    deferred_before = deferred.numpy().copy()

    torch_with = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    torch_with.grad = torch.tensor([1.0, 1.0])
    torch_def = torch.nn.Parameter(torch.tensor([10.0, 20.0]))

    mojo_opt = mojodl.optim.Adam([with_grad, deferred], lr=0.1)
    torch_opt = torch.optim.Adam([torch_with, torch_def], lr=0.1)

    mojo_opt.step()
    torch_opt.step()

    np.testing.assert_array_equal(deferred.numpy(), deferred_before)
    assert deferred.grad is None

    np.testing.assert_allclose(
        with_grad.numpy(),
        torch_with.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )

    # Second step: deferred now has a grad; its own step count starts at 1
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
# Meta: no raise, stays meta, no byte allocation across two Adam steps
# ---------------------------------------------------------------------------


def test_adam_step_on_meta_parameter_allocates_no_bytes():
    p = mojodl.nn.Parameter(mojodl.empty((3, 4), device="meta"))
    p.grad = mojodl.empty((3, 4), device="meta")

    opt = mojodl.optim.Adam([p])

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
# Invalid arguments (PyTorch-like)
# ---------------------------------------------------------------------------


def test_adam_negative_lr_raises_value_error():
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0], dtype=np.float32)))
    with pytest.raises(ValueError):
        mojodl.optim.Adam([p], lr=-1e-3)


def test_adam_negative_eps_raises_value_error():
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0], dtype=np.float32)))
    with pytest.raises(ValueError):
        mojodl.optim.Adam([p], eps=-1e-8)


def test_adam_beta1_out_of_range_raises_value_error():
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0], dtype=np.float32)))
    with pytest.raises(ValueError):
        mojodl.optim.Adam([p], betas=(1.0, 0.999))


def test_adam_beta2_out_of_range_raises_value_error():
    p = mojodl.nn.Parameter(mojodl.Tensor(np.array([1.0], dtype=np.float32)))
    with pytest.raises(ValueError):
        mojodl.optim.Adam([p], betas=(0.9, -0.1))


def test_adam_empty_param_list_raises_value_error():
    with pytest.raises(ValueError):
        mojodl.optim.Adam([])
