"""Acceptance: user-defined MLP matches a PyTorch twin end-to-end.

Covers named_parameters parity, forward / mse_loss / parameter grads after
loading torch weights, grad accumulation across a second fresh forward, and
Meta dry-run with no byte allocation. PyTorch CPU is the numerical oracle.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F

import mojodl


RTOL_FWD = 1e-5
ATOL_FWD = 1e-5
RTOL_LOSS = 1e-5
ATOL_LOSS = 1e-6
RTOL_GRAD = 1e-4
ATOL_GRAD = 1e-5


def _mojo_mlp(*, device: str = "cpu") -> mojodl.nn.Module:
    """User-defined MLP: Sequential body + head Linear."""

    class MLP(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = mojodl.nn.Sequential(
                mojodl.nn.Linear(8, 16, device=device),
                mojodl.nn.ReLU(),
                mojodl.nn.Linear(16, 16, device=device),
                mojodl.nn.ReLU(),
            )
            self.head = mojodl.nn.Linear(16, 4, device=device)

        def forward(self, x):
            return self.head(self.layers(x))

    return MLP()


def _torch_mlp() -> torch.nn.Module:
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 16),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(16, 4)

        def forward(self, x):
            return self.head(self.layers(x))

    return MLP()


def _named_shapes(module):
    return [(name, tuple(p.shape)) for name, p in module.named_parameters()]


def _load_torch_weights(mojo_model, torch_model):
    state = {
        k: v.detach().numpy() for k, v in torch_model.state_dict().items()
    }
    mojo_model.load_state_dict(state)


def _seeded_batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((32, 8)).astype(np.float32)
    y_np = rng.standard_normal((32, 4)).astype(np.float32)
    return x_np, y_np


# ---------------------------------------------------------------------------
# named_parameters parity
# ---------------------------------------------------------------------------


def test_mlp_named_parameters_names_and_shapes_match_torch():
    mojo_model = _mojo_mlp()
    torch_model = _torch_mlp()

    assert _named_shapes(mojo_model) == _named_shapes(torch_model)
    assert [n for n, _ in mojo_model.named_parameters()] == [
        "layers.0.weight",
        "layers.0.bias",
        "layers.2.weight",
        "layers.2.bias",
        "head.weight",
        "head.bias",
    ]
    assert _named_shapes(mojo_model) == [
        ("layers.0.weight", (16, 8)),
        ("layers.0.bias", (16,)),
        ("layers.2.weight", (16, 16)),
        ("layers.2.bias", (16,)),
        ("head.weight", (4, 16)),
        ("head.bias", (4,)),
    ]


# ---------------------------------------------------------------------------
# forward + mse_loss + parameter grads after loading torch weights
# ---------------------------------------------------------------------------


def test_mlp_forward_mse_loss_and_grads_match_torch():
    torch_model = _torch_mlp()
    mojo_model = _mojo_mlp()
    _load_torch_weights(mojo_model, torch_model)

    x_np, y_np = _seeded_batch(seed=0)

    x = mojodl.Tensor(x_np)
    y = mojodl.Tensor(y_np)
    out = mojo_model(x)
    loss = mojodl.mse_loss(out, y)
    loss.backward()

    torch_x = torch.from_numpy(x_np)
    torch_y = torch.from_numpy(y_np)
    torch_out = torch_model(torch_x)
    torch_loss = F.mse_loss(torch_out, torch_y)
    torch_loss.backward()

    np.testing.assert_allclose(
        out.numpy(),
        torch_out.detach().numpy(),
        rtol=RTOL_FWD,
        atol=ATOL_FWD,
    )
    np.testing.assert_allclose(
        loss.numpy(),
        torch_loss.detach().numpy(),
        rtol=RTOL_LOSS,
        atol=ATOL_LOSS,
    )

    torch_params = dict(torch_model.named_parameters())
    for name, param in mojo_model.named_parameters():
        assert isinstance(param.grad, mojodl.Tensor)
        np.testing.assert_allclose(
            param.grad.numpy(),
            torch_params[name].grad.numpy(),
            rtol=RTOL_GRAD,
            atol=ATOL_GRAD,
        )


# ---------------------------------------------------------------------------
# second fresh forward + backward accumulates (PyTorch semantics)
# ---------------------------------------------------------------------------


def test_mlp_second_backward_accumulates_matching_torch():
    torch_model = _torch_mlp()
    mojo_model = _mojo_mlp()
    _load_torch_weights(mojo_model, torch_model)

    x_np, y_np = _seeded_batch(seed=1)

    x = mojodl.Tensor(x_np)
    y = mojodl.Tensor(y_np)
    mojodl.mse_loss(mojo_model(x), y).backward()
    mojodl.mse_loss(mojo_model(x), y).backward()

    torch_x = torch.from_numpy(x_np)
    torch_y = torch.from_numpy(y_np)
    F.mse_loss(torch_model(torch_x), torch_y).backward()
    F.mse_loss(torch_model(torch_x), torch_y).backward()

    torch_params = dict(torch_model.named_parameters())
    for name, param in mojo_model.named_parameters():
        assert isinstance(param.grad, mojodl.Tensor)
        np.testing.assert_allclose(
            param.grad.numpy(),
            torch_params[name].grad.numpy(),
            rtol=RTOL_GRAD,
            atol=ATOL_GRAD,
        )


# ---------------------------------------------------------------------------
# Meta: forward + mse_loss + backward allocate no bytes
# ---------------------------------------------------------------------------


def test_mlp_meta_forward_backward_allocates_no_bytes():
    gc.collect()
    before = mojodl.memory_stats()["current_bytes"]

    model = _mojo_mlp(device="meta")
    x = mojodl.empty((32, 8), device="meta")
    y = mojodl.empty((32, 4), device="meta")
    loss = mojodl.mse_loss(model(x), y)

    assert isinstance(loss, mojodl.Tensor)
    assert loss.device == "meta"
    assert loss.shape == ()

    loss.backward()

    for _, param in model.named_parameters():
        assert isinstance(param.grad, mojodl.Tensor)
        assert param.grad.device == "meta"
        assert param.grad.shape == param.shape

    gc.collect()
    after = mojodl.memory_stats()["current_bytes"]
    assert after == before
