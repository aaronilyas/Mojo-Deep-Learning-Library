"""Tests for creating mojodl.Tensor from Python data and reading it back."""

import numpy as np
import pytest

import mojodl


def test_tensor_from_nested_list():
    t = mojodl.Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.dtype == mojodl.float32
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(t.numpy(), expected)
    assert t.numpy().dtype == np.float32


def test_tensor_from_numpy_float32_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 4, 5)).astype(np.float32)
    t = mojodl.Tensor(arr)
    np.testing.assert_array_equal(t.numpy(), arr)
    assert t.shape == (3, 4, 5)


@pytest.mark.parametrize(
    "arr",
    [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[1, 2], [3, 4]], dtype=np.int64),
    ],
    ids=["float64", "int64"],
)
def test_tensor_from_numpy_converts_to_float32(arr):
    t = mojodl.Tensor(arr)
    assert t.dtype == mojodl.float32


def test_tensor_from_scalar_list():
    t = mojodl.Tensor([1.5])
    assert isinstance(t.shape, tuple)
    assert all(isinstance(d, int) for d in t.shape)
    assert t.shape == (1,)
    np.testing.assert_array_equal(t.numpy(), np.array([1.5], dtype=np.float32))


def test_tensor_from_1d_list():
    t = mojodl.Tensor([1.0, 2.0, 3.0])
    assert isinstance(t.shape, tuple)
    assert all(isinstance(d, int) for d in t.shape)
    assert t.shape == (3,)
    np.testing.assert_array_equal(
        t.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )


def test_numpy_returns_a_copy():
    t = mojodl.Tensor([[1.0, 2.0], [3.0, 4.0]])
    a = t.numpy()
    a[0, 0] = 999.0
    b = t.numpy()
    assert b[0, 0] == 1.0


def test_ragged_nested_list_raises_value_error():
    with pytest.raises(ValueError):
        mojodl.Tensor([[1, 2], [3]])
