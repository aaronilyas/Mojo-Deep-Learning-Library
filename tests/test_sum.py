"""Tests for whole-tensor Tensor.sum() on cpu and Meta tensors.

PyTorch (CPU) is the numerical oracle. Axis / keepdims arguments are out of
scope here; this covers the no-argument form only.
"""

import gc

import numpy as np
import pytest
import torch

import mojodl


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (7,),
        (3, 4),
        (2, 3, 5),
    ],
    ids=["scalar", "1d", "2d", "3d"],
)
def test_sum_matches_torch(shape):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(shape).astype(np.float32)
    a = mojodl.Tensor(a_np)

    result = a.sum()
    expected = torch.from_numpy(a_np).sum().numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (2, 0),
    ],
    ids=["1d_empty", "2d_empty"],
)
def test_sum_empty_tensor_is_zero_scalar(shape):
    a = mojodl.Tensor(np.zeros(shape, dtype=np.float32))

    result = a.sum()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(result.numpy(), np.float32(0.0), rtol=1e-6, atol=1e-6)


def test_sum_does_not_mutate_input():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = mojodl.Tensor(a_np)

    result = a.sum()

    assert isinstance(result, mojodl.Tensor)
    assert result is not a
    np.testing.assert_array_equal(a.numpy(), a_np)


def test_sum_result_storage_is_accounted():
    a = mojodl.Tensor(np.zeros((64, 64), dtype=np.float32))
    baseline = mojodl.memory_stats()["current_bytes"]

    result = a.sum()

    assert mojodl.memory_stats()["current_bytes"] - baseline == 4
    del result
    gc.collect()


def test_sum_meta_returns_meta_scalar_without_allocating():
    t = mojodl.empty((3, 4), mojodl.float32, device="meta")
    before = mojodl.memory_stats()

    result = t.sum()

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    del t, result
    gc.collect()
