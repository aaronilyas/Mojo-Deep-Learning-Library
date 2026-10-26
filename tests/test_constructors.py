"""Tests for public constructors mojodl.zeros and mojodl.ones.

Constructors share empty()'s signature and argument validation:
zeros(shape, dtype=mojodl.float32, device="cpu", requires_grad=False)
(same for ones). NumPy is the value oracle. A Meta tensor (device "meta")
carries shape and dtype only and allocates nothing (GLOSSARY.md).

CONSTRUCTORS is a name table; constructors are resolved with getattr so a
missing attribute fails the test body, not collection. Extend later with
rand/randn using the same table.
"""

import gc
import math

import numpy as np
import pytest

import mojodl

# Name -> NumPy fill oracle. mojodl constructors resolved lazily via getattr.
CONSTRUCTORS = {
    "zeros": np.zeros,
    "ones": np.ones,
}

_CTOR_NAMES = list(CONSTRUCTORS.keys())

SHAPES = [(), (5,), (2, 3), (2, 0, 3)]


def _ctor(name: str):
    return getattr(mojodl, name)


def _numel(shape: tuple) -> int:
    return math.prod(shape) if shape else 1


@pytest.mark.parametrize("name", _CTOR_NAMES)
@pytest.mark.parametrize("shape", SHAPES, ids=["scalar", "1d", "2d", "zero_dim"])
def test_constructor_values_match_numpy(name, shape):
    t = _ctor(name)(shape)
    expected = CONSTRUCTORS[name](shape, dtype=np.float32)
    np.testing.assert_array_equal(t.numpy(), expected)


@pytest.mark.parametrize("name", _CTOR_NAMES)
@pytest.mark.parametrize("shape", SHAPES, ids=["scalar", "1d", "2d", "zero_dim"])
def test_constructor_returns_cpu_float32_tensor_and_allocates(name, shape):
    baseline = mojodl.memory_stats()["current_bytes"]

    t = _ctor(name)(shape)

    assert isinstance(t, mojodl.Tensor)
    assert t.shape == shape
    assert t.dtype == mojodl.float32
    assert t.device == "cpu"
    assert mojodl.memory_stats()["current_bytes"] - baseline == 4 * _numel(shape)
    del t
    gc.collect()


@pytest.mark.parametrize("name", _CTOR_NAMES)
def test_constructor_meta_returns_shape_dtype_without_allocating(name):
    before = mojodl.memory_stats()

    t = _ctor(name)((4096, 4096), device="meta")

    after = mojodl.memory_stats()
    assert isinstance(t, mojodl.Tensor)
    assert t.shape == (4096, 4096)
    assert t.dtype == mojodl.float32
    assert t.device == "meta"
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    del t
    gc.collect()


@pytest.mark.parametrize("name", _CTOR_NAMES)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_constructor_requires_grad_default_false(name, device):
    t = _ctor(name)((2, 2), device=device)

    assert t.requires_grad is False
    assert t.is_leaf is True
    assert t.grad is None


@pytest.mark.parametrize("name", _CTOR_NAMES)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_constructor_requires_grad_true_is_leaf(name, device):
    t = _ctor(name)((2, 2), device=device, requires_grad=True)

    assert t.requires_grad is True
    assert t.is_leaf is True
    assert t.grad is None
    assert t.device == device


@pytest.mark.parametrize("name", _CTOR_NAMES)
def test_constructor_non_float32_dtype_raises_type_error(name):
    with pytest.raises(TypeError):
        _ctor(name)((2, 3), dtype="float64")


@pytest.mark.parametrize("name", _CTOR_NAMES)
def test_constructor_unknown_device_raises_value_error(name):
    with pytest.raises(ValueError):
        _ctor(name)((2, 3), device="cuda")


@pytest.mark.parametrize("name", _CTOR_NAMES)
def test_constructor_negative_dim_raises_value_error(name):
    with pytest.raises(ValueError):
        _ctor(name)((2, -1))


@pytest.mark.parametrize("name", _CTOR_NAMES)
def test_constructor_non_int_dim_raises_type_error(name):
    with pytest.raises(TypeError):
        _ctor(name)((2.5,))
