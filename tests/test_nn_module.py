"""Tests for mojodl.nn.Module attribute registration, Parameter, and __call__.

PyTorch (CPU) is the behavioural oracle for named_parameters order, dedup,
and Parameter leaf/grad defaults. Linear / ReLU / Sequential, state dict,
and losses are out of scope here.
"""

import numpy as np
import pytest
import torch

import mojodl


# ---------------------------------------------------------------------------
# Twin builders: same attribute names and assignment order on both sides
# ---------------------------------------------------------------------------


def _build_nested_mojodl():
    """Outer Module with own Parameter + two inner sub-modules; one nests deeper."""

    class InnerMost(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = mojodl.nn.Parameter(mojodl.zeros((3,)))

    class Encoder(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mojodl.nn.Parameter(mojodl.ones((2, 3)))
            self.scale = mojodl.nn.Parameter(mojodl.ones((2,)))
            self.inner = InnerMost()

    class Decoder(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mojodl.nn.Parameter(mojodl.ones((4, 2)))
            self.bias = mojodl.nn.Parameter(mojodl.zeros((4,)))

    class Model(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = mojodl.nn.Parameter(mojodl.ones((5,)))
            self.encoder = Encoder()
            self.decoder = Decoder()

    return Model()


def _build_nested_torch():
    class InnerMost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(3))

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 3))
            self.scale = torch.nn.Parameter(torch.ones(2))
            self.inner = InnerMost()

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 2))
            self.bias = torch.nn.Parameter(torch.zeros(4))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Parameter(torch.ones(5))
            self.encoder = Encoder()
            self.decoder = Decoder()

    return Model()


def _named_shapes(module):
    return [(name, tuple(p.shape)) for name, p in module.named_parameters()]


# ---------------------------------------------------------------------------
# named_parameters / parameters / nesting
# ---------------------------------------------------------------------------


def test_nested_named_parameters_match_torch_order_and_shapes():
    mojo_model = _build_nested_mojodl()
    torch_model = _build_nested_torch()

    assert _named_shapes(mojo_model) == _named_shapes(torch_model)
    assert _named_shapes(mojo_model) == [
        ("head", (5,)),
        ("encoder.weight", (2, 3)),
        ("encoder.scale", (2,)),
        ("encoder.inner.bias", (3,)),
        ("decoder.weight", (4, 2)),
        ("decoder.bias", (4,)),
    ]


def test_parameters_are_named_parameters_values_by_identity():
    model = _build_nested_mojodl()

    named_values = [p for _, p in model.named_parameters()]
    params = list(model.parameters())

    assert len(params) == len(named_values)
    for a, b in zip(params, named_values):
        assert a is b


def test_registered_parameters_and_submodules_readable_as_attributes():
    model = _build_nested_mojodl()

    named = dict(model.named_parameters())
    assert model.head is named["head"]
    assert model.encoder.weight is named["encoder.weight"]
    assert model.encoder.scale is named["encoder.scale"]
    assert model.encoder.inner.bias is named["encoder.inner.bias"]
    assert model.decoder.weight is named["decoder.weight"]
    assert model.decoder.bias is named["decoder.bias"]
    assert model.encoder.inner is model.encoder.inner
    assert isinstance(model.encoder, mojodl.nn.Module)
    assert isinstance(model.decoder, mojodl.nn.Module)
    assert isinstance(model.encoder.inner, mojodl.nn.Module)


# ---------------------------------------------------------------------------
# Parameter type / leaf / requires_grad / values
# ---------------------------------------------------------------------------


def test_parameter_is_tensor_leaf_requiring_grad_by_default():
    t = mojodl.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    p = mojodl.nn.Parameter(t)

    assert isinstance(p, mojodl.Tensor)
    assert isinstance(p, mojodl.nn.Parameter)
    assert p.is_leaf is True
    assert p.requires_grad is True
    assert p.grad is None


def test_parameter_requires_grad_false():
    t = mojodl.Tensor(np.array([1.0, 2.0], dtype=np.float32))
    p = mojodl.nn.Parameter(t, requires_grad=False)

    assert p.requires_grad is False
    assert p.is_leaf is True
    assert p.grad is None


def test_parameter_from_tensor_preserves_shape_and_values():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    p = mojodl.nn.Parameter(mojodl.Tensor(arr))

    assert p.shape == arr.shape
    np.testing.assert_array_equal(p.numpy(), arr)


# ---------------------------------------------------------------------------
# Non-Parameter attributes are not registered
# ---------------------------------------------------------------------------


def test_plain_tensor_and_python_attrs_not_registered_as_parameters():
    class Model(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer = mojodl.Tensor(
                np.array([1.0, 2.0], dtype=np.float32)
            )
            self.weight = mojodl.nn.Parameter(
                mojodl.Tensor(np.array([3.0, 4.0, 5.0], dtype=np.float32))
            )
            self.count = 7
            self.tags = ["a", "b"]

    model = Model()

    assert _named_shapes(model) == [("weight", (3,))]
    assert isinstance(model.buffer, mojodl.Tensor)
    np.testing.assert_array_equal(
        model.buffer.numpy(), np.array([1.0, 2.0], dtype=np.float32)
    )
    assert model.count == 7
    assert model.tags == ["a", "b"]


# ---------------------------------------------------------------------------
# Re-assignment and dedup (PyTorch semantics)
# ---------------------------------------------------------------------------


def test_reassigning_parameter_attribute_yields_new_object_once():
    class Model(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mojodl.nn.Parameter(mojodl.ones((2,)))

    model = Model()
    first = model.w
    model.w = mojodl.nn.Parameter(mojodl.zeros((2,)))
    named = list(model.named_parameters())

    assert len(named) == 1
    name, p = named[0]
    assert name == "w"
    assert p is model.w
    assert p is not first
    np.testing.assert_array_equal(p.numpy(), np.zeros(2, dtype=np.float32))

    class TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(2))

    torch_model = TorchModel()
    torch_model.w = torch.nn.Parameter(torch.zeros(2))
    assert [n for n, _ in named] == [n for n, _ in torch_model.named_parameters()]


def test_same_parameter_two_attributes_yielded_once_under_first_name():
    class Model(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            p = mojodl.nn.Parameter(mojodl.ones((2,)))
            self.a = p
            self.b = p

    model = Model()
    named = list(model.named_parameters())

    assert len(named) == 1
    assert named[0][0] == "a"
    assert named[0][1] is model.a
    assert model.a is model.b
    assert list(model.parameters()) == [model.a]


def test_same_parameter_shared_across_submodules_yielded_once_under_first_name():
    class Block(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mojodl.nn.Parameter(mojodl.ones((2, 3)))
            self.bias = mojodl.nn.Parameter(mojodl.zeros((2,)))

    class Model(mojodl.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = Block()
            self.right = Block()
            self.right.weight = self.left.weight

    mojo_model = Model()

    class TorchBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 3))
            self.bias = torch.nn.Parameter(torch.zeros(2))

    class TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = TorchBlock()
            self.right = TorchBlock()
            self.right.weight = self.left.weight

    torch_model = TorchModel()

    assert _named_shapes(mojo_model) == _named_shapes(torch_model)
    assert [n for n, _ in mojo_model.named_parameters()] == [
        "left.weight",
        "left.bias",
        "right.bias",
    ]
    assert mojo_model.right.weight is mojo_model.left.weight


# ---------------------------------------------------------------------------
# __call__ → forward
# ---------------------------------------------------------------------------


def test_calling_module_forwards_args_to_forward():
    class AddBias(mojodl.nn.Module):
        def forward(self, x, bias=None):
            if bias is None:
                return x
            return x + bias

    model = AddBias()
    x = mojodl.Tensor(np.array([1.0, 2.0], dtype=np.float32))
    bias = mojodl.Tensor(np.array([10.0, 20.0], dtype=np.float32))

    out_pos = model(x, bias)
    out_kw = model(x, bias=bias)
    out_default = model(x)

    np.testing.assert_array_equal(
        out_pos.numpy(), np.array([11.0, 22.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out_kw.numpy(), np.array([11.0, 22.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out_default.numpy(), np.array([1.0, 2.0], dtype=np.float32)
    )


def test_calling_module_without_forward_raises_not_implemented():
    class Empty(mojodl.nn.Module):
        pass

    model = Empty()
    x = mojodl.Tensor(np.array([1.0], dtype=np.float32))

    with pytest.raises(NotImplementedError):
        model(x)


# ---------------------------------------------------------------------------
# Autograd through a Parameter
# ---------------------------------------------------------------------------


def test_backward_through_parameter_fills_grad_matching_torch():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    w_np = np.array([[0.5, -1.0], [2.0, 0.0]], dtype=np.float32)

    class Scale(mojodl.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = mojodl.nn.Parameter(mojodl.Tensor(w))

        def forward(self, x):
            return (self.w * x).sum()

    model = Scale(w_np)
    x = mojodl.Tensor(x_np)
    model(x).backward()

    class TorchScale(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(w))

        def forward(self, x):
            return (self.w * x).sum()

    torch_model = TorchScale(w_np)
    torch_x = torch.tensor(x_np)
    torch_model(torch_x).backward()

    assert isinstance(model.w.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        model.w.grad.numpy(),
        torch_model.w.grad.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        model.w.grad.numpy(),
        x_np,
        rtol=1e-6,
        atol=1e-6,
    )
