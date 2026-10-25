"""Tests for same-shape Tensor + Tensor and ShapeError on mismatch.

PyTorch (CPU) is the numerical oracle. Broadcasting is out of scope here.
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
def test_add_same_shape_matches_torch(shape):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(shape).astype(np.float32)
    b_np = rng.standard_normal(shape).astype(np.float32)
    a = mojodl.Tensor(a_np)
    b = mojodl.Tensor(b_np)

    result = a + b
    expected = (torch.from_numpy(a_np) + torch.from_numpy(b_np)).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=0)


def test_add_does_not_mutate_inputs_and_returns_new_tensor():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    a = mojodl.Tensor(a_np)
    b = mojodl.Tensor(b_np)

    result = a + b

    assert isinstance(result, mojodl.Tensor)
    assert result is not a
    assert result is not b
    np.testing.assert_array_equal(a.numpy(), a_np)
    np.testing.assert_array_equal(b.numpy(), b_np)


def test_add_result_storage_is_accounted():
    a = mojodl.Tensor(np.zeros((64, 64), dtype=np.float32))
    b = mojodl.Tensor(np.zeros((64, 64), dtype=np.float32))
    nbytes = 64 * 64 * 4
    baseline = mojodl.memory_stats()["current_bytes"]

    result = a + b

    assert mojodl.memory_stats()["current_bytes"] - baseline == nbytes
    del result
    gc.collect()


def test_add_shape_mismatch_raises_shape_error():
    a = mojodl.Tensor(np.zeros((2, 3), dtype=np.float32))
    b = mojodl.Tensor(np.zeros((3, 2), dtype=np.float32))

    assert issubclass(mojodl.ShapeError, ValueError)

    with pytest.raises(mojodl.ShapeError) as exc_info:
        a + b

    msg = str(exc_info.value)
    assert "(2, 3)" in msg
    assert "(3, 2)" in msg
