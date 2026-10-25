"""Tests for Meta tensors via mojodl.empty(..., device="meta").

A Meta tensor carries shape and dtype but no data (GLOSSARY.md). These tests
drive only the public Python API; production code for empty/device is not yet
implemented, so they fail with AttributeError until it lands.
"""

import gc

import pytest

import mojodl


def test_empty_meta_returns_tensor_with_shape_dtype_device():
    t = mojodl.empty((2, 3), mojodl.float32, device="meta")

    assert isinstance(t, mojodl.Tensor)
    assert t.shape == (2, 3)
    assert t.dtype == mojodl.float32
    assert t.device == "meta"


def test_large_meta_tensor_does_not_allocate_bytes():
    before = mojodl.memory_stats()

    t = mojodl.empty((4096, 4096), mojodl.float32, device="meta")

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    assert t.device == "meta"
    del t
    gc.collect()


def test_add_meta_tensors_same_shape_returns_meta_without_allocating():
    a = mojodl.empty((3, 4), mojodl.float32, device="meta")
    b = mojodl.empty((3, 4), mojodl.float32, device="meta")
    baseline = mojodl.memory_stats()["current_bytes"]

    result = a + b

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == (3, 4)
    assert result.dtype == mojodl.float32
    assert mojodl.memory_stats()["current_bytes"] == baseline


def test_add_meta_tensors_shape_mismatch_raises_shape_error():
    a = mojodl.empty((2, 3), mojodl.float32, device="meta")
    b = mojodl.empty((3, 2), mojodl.float32, device="meta")

    with pytest.raises(mojodl.ShapeError):
        a + b


def test_ordinary_tensor_device_is_cpu():
    t = mojodl.Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.device == "cpu"


def test_empty_defaults_to_cpu_and_allocates_bytes():
    baseline = mojodl.memory_stats()["current_bytes"]

    t = mojodl.empty((2, 2), mojodl.float32)

    assert isinstance(t, mojodl.Tensor)
    assert t.device == "cpu"
    assert t.shape == (2, 2)
    assert t.dtype == mojodl.float32
    assert mojodl.memory_stats()["current_bytes"] - baseline == 16
    del t
    gc.collect()
