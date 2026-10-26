"""Tests for Module.state_dict / load_state_dict NumPy export and import.

PyTorch (CPU) is the behavioural oracle for qualified parameter names and for
forward parity after loading a torch twin's weights. state_dict / load_state_dict
are not implemented yet, so these tests fail with AttributeError until that
slice lands.
"""

from __future__ import annotations

import numpy as np
import torch

import mojodl


RTOL = 1e-5
ATOL = 1e-5


def _mlp(seed: int):
    """Sequential(Linear(4, 8), ReLU(), Linear(8, 3)) with distinct Linear seeds."""
    return mojodl.nn.Sequential(
        mojodl.nn.Linear(4, 8, seed=seed),
        mojodl.nn.ReLU(),
        mojodl.nn.Linear(8, 3, seed=seed + 10),
    )


def _torch_mlp():
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )


def _user_net(seed: int):
    """User Module wrapping the MLP Sequential plus a head Linear."""

    class Net(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = mojodl.nn.Sequential(
                mojodl.nn.Linear(4, 8, seed=seed),
                mojodl.nn.ReLU(),
                mojodl.nn.Linear(8, 3, seed=seed + 10),
            )
            self.head = mojodl.nn.Linear(3, 2, seed=seed + 20)

        def forward(self, x):
            return self.head(self.layers(x))

    return Net()


def _torch_user_net():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 3),
            )
            self.head = torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.head(self.layers(x))

    return Net()


# ---------------------------------------------------------------------------
# state_dict: keys, dtypes, shapes, values
# ---------------------------------------------------------------------------


def test_mlp_state_dict_keys_dtypes_shapes_and_values():
    model = _mlp(seed=0)
    torch_model = _torch_mlp()

    named = [n for n, _ in model.named_parameters()]
    state = model.state_dict()

    assert list(state.keys()) == named
    assert list(state.keys()) == list(torch_model.state_dict().keys())
    assert named == ["0.weight", "0.bias", "2.weight", "2.bias"]

    for name, param in model.named_parameters():
        arr = state[name]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == tuple(param.shape)
        np.testing.assert_array_equal(arr, param.numpy())


def test_user_module_state_dict_keys_dtypes_shapes_and_values():
    model = _user_net(seed=0)
    torch_model = _torch_user_net()

    named = [n for n, _ in model.named_parameters()]
    state = model.state_dict()

    assert list(state.keys()) == named
    assert list(state.keys()) == list(torch_model.state_dict().keys())
    assert named == [
        "layers.0.weight",
        "layers.0.bias",
        "layers.2.weight",
        "layers.2.bias",
        "head.weight",
        "head.bias",
    ]

    for name, param in model.named_parameters():
        arr = state[name]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == tuple(param.shape)
        np.testing.assert_array_equal(arr, param.numpy())


def test_state_dict_returns_independent_copies():
    model = _mlp(seed=0)
    state = model.state_dict()
    originals = {k: v.copy() for k, v in state.items()}

    for arr in state.values():
        arr += 1.0

    for name, param in model.named_parameters():
        np.testing.assert_array_equal(param.numpy(), originals[name])


# ---------------------------------------------------------------------------
# load_state_dict: round trip, identity, copy semantics, dtype
# ---------------------------------------------------------------------------


def test_load_state_dict_round_trip_arrays_and_forward():
    model_a = _mlp(seed=0)
    model_b = _mlp(seed=1)

    state_a = model_a.state_dict()
    model_b.load_state_dict(state_a)
    state_b = model_b.state_dict()

    assert list(state_b.keys()) == list(state_a.keys())
    for key in state_a:
        np.testing.assert_array_equal(state_b[key], state_a[key])

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((6, 4)).astype(np.float32)
    out_a = model_a(mojodl.Tensor(x_np))
    out_b = model_b(mojodl.Tensor(x_np))
    np.testing.assert_array_equal(out_b.numpy(), out_a.numpy())


def test_load_state_dict_preserves_parameter_identity():
    model_a = _mlp(seed=0)
    model_b = _mlp(seed=1)

    params_before = list(model_b.parameters())
    assert len(params_before) > 0

    # Confirm seeds differ so the load actually changes values.
    assert not np.array_equal(
        params_before[0].numpy(), list(model_a.parameters())[0].numpy()
    )

    model_b.load_state_dict(model_a.state_dict())
    params_after = list(model_b.parameters())

    assert len(params_after) == len(params_before)
    for before, after in zip(params_before, params_after):
        assert before is after

    for (_, pa), (_, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        np.testing.assert_array_equal(pb.numpy(), pa.numpy())


def test_load_state_dict_copies_from_source_arrays():
    model_a = _mlp(seed=0)
    model_b = _mlp(seed=1)

    source = {k: v.copy() for k, v in model_a.state_dict().items()}
    model_b.load_state_dict(source)

    for arr in source.values():
        arr += 99.0

    for name, param in model_b.named_parameters():
        np.testing.assert_array_equal(
            param.numpy(), model_a.state_dict()[name]
        )


def test_load_state_dict_accepts_float64_and_stores_float32_cast():
    model_a = _mlp(seed=0)
    model_b = _mlp(seed=1)

    source = {
        k: v.astype(np.float64) for k, v in model_a.state_dict().items()
    }
    assert all(arr.dtype == np.float64 for arr in source.values())

    model_b.load_state_dict(source)

    for name, param in model_b.named_parameters():
        got = param.numpy()
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, source[name].astype(np.float32))


# ---------------------------------------------------------------------------
# load_state_dict from a torch twin → forward parity
# ---------------------------------------------------------------------------


def test_load_torch_state_dict_makes_mlp_forward_match_torch():
    torch_model = _torch_mlp()
    model = _mlp(seed=0)

    state = {
        k: v.detach().numpy() for k, v in torch_model.state_dict().items()
    }
    model.load_state_dict(state)

    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((6, 4)).astype(np.float32)

    out = model(mojodl.Tensor(x_np))
    torch_out = torch_model(torch.from_numpy(x_np))

    np.testing.assert_allclose(
        out.numpy(),
        torch_out.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_load_torch_state_dict_makes_user_module_forward_match_torch():
    torch_model = _torch_user_net()
    model = _user_net(seed=0)

    state = {
        k: v.detach().numpy() for k, v in torch_model.state_dict().items()
    }
    model.load_state_dict(state)

    rng = np.random.default_rng(12)
    x_np = rng.standard_normal((6, 4)).astype(np.float32)

    out = model(mojodl.Tensor(x_np))
    torch_out = torch_model(torch.from_numpy(x_np))

    np.testing.assert_allclose(
        out.numpy(),
        torch_out.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
