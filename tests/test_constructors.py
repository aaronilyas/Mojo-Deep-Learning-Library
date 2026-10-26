"""Tests for public constructors mojodl.zeros, ones, rand, and randn.

Constructors share empty()'s signature and argument validation:
zeros(shape, dtype=mojodl.float32, device="cpu", requires_grad=False)
(same for ones / rand / randn). rand and randn also accept seed=None.
NumPy is the value oracle for zeros/ones. A Meta tensor (device "meta")
carries shape and dtype only and allocates nothing (GLOSSARY.md).

FILL_CONSTRUCTORS is the exact-fill name table (zeros/ones). _CTOR_NAMES
covers all four constructors for shared shape/dtype/device/requires_grad
and validation tests. Constructors are resolved with getattr so a missing
attribute fails the test body, not collection.
"""

import gc
import math

import numpy as np
import pytest

import mojodl

# Name -> NumPy fill oracle. Exact-value tests use this table only.
FILL_CONSTRUCTORS = {
    "zeros": np.zeros,
    "ones": np.ones,
}

# All public constructors that share empty()'s validation and Meta behaviour.
_CTOR_NAMES = ["zeros", "ones", "rand", "randn"]
_RNG_NAMES = ["rand", "randn"]

SHAPES = [(), (5,), (2, 3), (2, 0, 3)]


def _ctor(name: str):
    return getattr(mojodl, name)


def _numel(shape: tuple) -> int:
    return math.prod(shape) if shape else 1


@pytest.mark.parametrize("name", list(FILL_CONSTRUCTORS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=["scalar", "1d", "2d", "zero_dim"])
def test_constructor_values_match_numpy(name, shape):
    t = _ctor(name)(shape)
    expected = FILL_CONSTRUCTORS[name](shape, dtype=np.float32)
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


# --- rand / randn: distributional properties and seed -------------------


def test_rand_uniform_on_unit_interval():
    t = _ctor("rand")((10000,))
    arr = t.numpy()
    assert arr.shape == (10000,)
    assert np.all(arr >= 0.0) and np.all(arr < 1.0)
    assert abs(float(arr.mean()) - 0.5) < 0.02
    assert len(np.unique(arr)) >= 9000


def test_randn_standard_normal():
    t = _ctor("randn")((10000,))
    arr = t.numpy()
    assert arr.shape == (10000,)
    assert np.all(np.isfinite(arr))
    assert abs(float(arr.mean())) < 0.05
    assert abs(float(arr.std()) - 1.0) < 0.05
    assert len(np.unique(arr)) >= 9000


@pytest.mark.parametrize("name", _RNG_NAMES)
def test_rng_seed_reproducible_and_distinct(name):
    f = _ctor(name)
    shape = (3, 4, 5)

    a = f(shape, seed=123)
    b = f(shape, seed=123)
    np.testing.assert_array_equal(a.numpy(), b.numpy())

    c = f(shape, seed=124)
    assert not np.array_equal(a.numpy(), c.numpy())

    d = f(shape)
    e = f(shape)
    assert not np.array_equal(d.numpy(), e.numpy())


@pytest.mark.parametrize("name", _RNG_NAMES)
def test_rng_seed_on_meta_allocates_nothing(name):
    before = mojodl.memory_stats()

    t = _ctor(name)((64, 64), device="meta", seed=0)

    after = mojodl.memory_stats()
    assert isinstance(t, mojodl.Tensor)
    assert t.shape == (64, 64)
    assert t.dtype == mojodl.float32
    assert t.device == "meta"
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    del t
    gc.collect()


@pytest.mark.parametrize("name", _RNG_NAMES)
@pytest.mark.parametrize(
    "seed,exc",
    [
        (1.5, TypeError),
        ("x", TypeError),
        (-1, ValueError),
    ],
    ids=["float", "str", "negative"],
)
def test_rng_invalid_seed_raises(name, seed, exc):
    with pytest.raises(exc):
        _ctor(name)((2, 3), seed=seed)
