"""Tests for mojodl.nn.Linear, ReLU, and Sequential against PyTorch twins.

PyTorch (CPU) is the behavioural and numerical oracle. Weight state is copied
in by assigning Parameters from numpy (state_dict import is a later slice).
Linear / ReLU / Sequential are not implemented yet, so these tests fail with
AttributeError until that slice lands.
"""

from __future__ import annotations

import gc
import math

import numpy as np
import torch

import mojodl


RTOL = 1e-5
ATOL = 1e-5


def _named_shapes(module):
    return [(name, tuple(p.shape)) for name, p in module.named_parameters()]


def _named_module_names(module):
    return [name for name, _ in module.named_modules()]


def _copy_linear_weights(mojo_lin, torch_lin):
    """Copy torch Linear weight/bias into a mojodl Linear via Parameter assign."""
    mojo_lin.weight = mojodl.nn.Parameter(
        mojodl.Tensor(torch_lin.weight.detach().numpy().copy())
    )
    if torch_lin.bias is not None:
        mojo_lin.bias = mojodl.nn.Parameter(
            mojodl.Tensor(torch_lin.bias.detach().numpy().copy())
        )


# ---------------------------------------------------------------------------
# Sequential / nested Module: named_parameters and named_modules
# ---------------------------------------------------------------------------


def test_sequential_named_parameters_and_modules_match_torch():
    mojo_seq = mojodl.nn.Sequential(
        mojodl.nn.Linear(4, 8),
        mojodl.nn.ReLU(),
        mojodl.nn.Linear(8, 3),
    )
    torch_seq = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )

    assert _named_shapes(mojo_seq) == _named_shapes(torch_seq)
    assert _named_shapes(mojo_seq) == [
        ("0.weight", (8, 4)),
        ("0.bias", (8,)),
        ("2.weight", (3, 8)),
        ("2.bias", (3,)),
    ]
    assert _named_module_names(mojo_seq) == _named_module_names(torch_seq)
    assert _named_module_names(mojo_seq) == ["", "0", "1", "2"]


def test_user_module_with_sequential_and_head_matches_torch_names():
    class MojoNet(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = mojodl.nn.Sequential(
                mojodl.nn.Linear(4, 8),
                mojodl.nn.ReLU(),
                mojodl.nn.Linear(8, 6),
                mojodl.nn.ReLU(),
            )
            self.head = mojodl.nn.Linear(6, 3)

    class TorchNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 6),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(6, 3)

    mojo_net = MojoNet()
    torch_net = TorchNet()

    assert _named_shapes(mojo_net) == _named_shapes(torch_net)
    assert [n for n, _ in mojo_net.named_parameters()] == [
        "layers.0.weight",
        "layers.0.bias",
        "layers.2.weight",
        "layers.2.bias",
        "head.weight",
        "head.bias",
    ]
    assert "head.bias" in dict(mojo_net.named_parameters())
    assert _named_module_names(mojo_net) == _named_module_names(torch_net)


# ---------------------------------------------------------------------------
# Linear: bias=False, initialisation, Parameter leaf properties
# ---------------------------------------------------------------------------


def test_linear_bias_false_has_only_weight_parameter():
    lin = mojodl.nn.Linear(4, 3, bias=False)
    torch_lin = torch.nn.Linear(4, 3, bias=False)

    assert _named_shapes(lin) == _named_shapes(torch_lin)
    assert _named_shapes(lin) == [("weight", (3, 4))]
    assert lin.bias is None
    assert torch_lin.bias is None


def test_linear_init_values_lie_in_pytorch_default_bound():
    in_features = 16
    out_features = 8
    bound = 1.0 / math.sqrt(in_features)

    lin = mojodl.nn.Linear(in_features, out_features)
    w = lin.weight.numpy()
    b = lin.bias.numpy()

    assert w.shape == (out_features, in_features)
    assert b.shape == (out_features,)
    assert np.all(w >= -bound) and np.all(w <= bound)
    assert np.all(b >= -bound) and np.all(b <= bound)


def test_linear_init_large_layer_is_non_constant_and_fills_bound():
    in_features = 256
    out_features = 128
    bound = 1.0 / math.sqrt(in_features)

    lin = mojodl.nn.Linear(in_features, out_features)
    w = lin.weight.numpy()
    b = lin.bias.numpy()

    assert not np.allclose(w, w.flat[0])
    assert not np.allclose(b, b.flat[0])
    assert float(np.max(np.abs(w))) > 0.5 * bound


def test_linear_same_seed_produces_bit_identical_weight_and_bias():
    a = mojodl.nn.Linear(4, 3, seed=42)
    b = mojodl.nn.Linear(4, 3, seed=42)

    np.testing.assert_array_equal(a.weight.numpy(), b.weight.numpy())
    np.testing.assert_array_equal(a.bias.numpy(), b.bias.numpy())


def test_linear_different_seeds_produce_different_weights():
    a = mojodl.nn.Linear(4, 3, seed=1)
    b = mojodl.nn.Linear(4, 3, seed=2)

    assert not np.array_equal(a.weight.numpy(), b.weight.numpy())


def test_linear_weight_and_bias_are_parameters_requiring_grad_and_leaves():
    lin = mojodl.nn.Linear(4, 3)

    assert isinstance(lin.weight, mojodl.nn.Parameter)
    assert isinstance(lin.bias, mojodl.nn.Parameter)
    assert lin.weight.requires_grad is True
    assert lin.bias.requires_grad is True
    assert lin.weight.is_leaf is True
    assert lin.bias.is_leaf is True
    assert lin.weight.grad_fn is None
    assert lin.bias.grad_fn is None


# ---------------------------------------------------------------------------
# Linear: forward and backward vs torch (copied weights)
# ---------------------------------------------------------------------------


def test_linear_forward_matches_torch_with_copied_weights():
    torch_lin = torch.nn.Linear(4, 3)
    lin = mojodl.nn.Linear(4, 3)
    _copy_linear_weights(lin, torch_lin)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((5, 4)).astype(np.float32)

    out = lin(mojodl.Tensor(x_np))
    torch_out = torch_lin(torch.from_numpy(x_np))

    assert out.shape == (5, 3)
    assert tuple(torch_out.shape) == (5, 3)
    np.testing.assert_allclose(
        out.numpy(), torch_out.detach().numpy(), rtol=RTOL, atol=ATOL
    )


def test_linear_forward_bias_false_matches_torch_with_copied_weights():
    torch_lin = torch.nn.Linear(4, 3, bias=False)
    lin = mojodl.nn.Linear(4, 3, bias=False)
    _copy_linear_weights(lin, torch_lin)

    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((5, 4)).astype(np.float32)

    out = lin(mojodl.Tensor(x_np))
    torch_out = torch_lin(torch.from_numpy(x_np))

    assert out.shape == (5, 3)
    np.testing.assert_allclose(
        out.numpy(), torch_out.detach().numpy(), rtol=RTOL, atol=ATOL
    )


def test_linear_backward_matches_torch_weight_bias_and_input_grads():
    torch_lin = torch.nn.Linear(4, 3)
    lin = mojodl.nn.Linear(4, 3)
    _copy_linear_weights(lin, torch_lin)

    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((5, 4)).astype(np.float32)

    x = mojodl.Tensor(x_np, requires_grad=True)
    out = lin(x)
    out.sum().backward()

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_out = torch_lin(torch_x)
    torch_out.sum().backward()

    np.testing.assert_allclose(
        lin.weight.grad.numpy(),
        torch_lin.weight.grad.numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        lin.bias.grad.numpy(),
        torch_lin.bias.grad.numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        x.grad.numpy(),
        torch_x.grad.numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# ReLU module
# ---------------------------------------------------------------------------


def test_relu_module_has_no_parameters():
    relu = mojodl.nn.ReLU()
    assert list(relu.named_parameters()) == []
    assert list(relu.parameters()) == []


def test_relu_module_forward_matches_torch_on_mixed_sign_input():
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((4, 5)).astype(np.float32)
    # Ensure mixed signs so the ReLU kink is exercised.
    x_np[0, 0] = -1.5
    x_np[1, 1] = 0.0
    x_np[2, 2] = 2.25

    relu = mojodl.nn.ReLU()
    out = relu(mojodl.Tensor(x_np))
    expected = torch.relu(torch.from_numpy(x_np)).numpy()

    assert out.shape == (4, 5)
    np.testing.assert_allclose(out.numpy(), expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Sequential: container protocol and forward
# ---------------------------------------------------------------------------


def test_sequential_len_indexing_and_iteration():
    lin0 = mojodl.nn.Linear(4, 8)
    relu = mojodl.nn.ReLU()
    lin1 = mojodl.nn.Linear(8, 3)
    seq = mojodl.nn.Sequential(lin0, relu, lin1)

    assert len(seq) == 3
    assert seq[0] is lin0
    assert seq[-1] is lin1
    assert list(seq) == [lin0, relu, lin1]


def test_sequential_forward_matches_torch_with_copied_weights():
    torch_seq = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )
    seq = mojodl.nn.Sequential(
        mojodl.nn.Linear(4, 8),
        mojodl.nn.ReLU(),
        mojodl.nn.Linear(8, 3),
    )
    _copy_linear_weights(seq[0], torch_seq[0])
    _copy_linear_weights(seq[2], torch_seq[2])

    rng = np.random.default_rng(4)
    x_np = rng.standard_normal((5, 4)).astype(np.float32)

    out = seq(mojodl.Tensor(x_np))
    torch_out = torch_seq(torch.from_numpy(x_np))

    assert out.shape == (5, 3)
    np.testing.assert_allclose(
        out.numpy(), torch_out.detach().numpy(), rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# Meta device
# ---------------------------------------------------------------------------


def test_linear_meta_does_not_allocate_and_forward_stays_meta():
    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    lin = mojodl.nn.Linear(64, 32, device="meta")

    assert mojodl.memory_stats()["current_bytes"] == baseline
    assert lin.weight.device == "meta"
    assert lin.bias.device == "meta"

    x = mojodl.empty((5, 64), device="meta")
    out = lin(x)

    assert out.device == "meta"
    assert out.shape == (5, 32)
    assert mojodl.memory_stats()["current_bytes"] == baseline
