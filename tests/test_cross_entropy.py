"""Tests for mojodl.cross_entropy(logits, labels) forward value and Meta behaviour.

PyTorch (CPU) is the numerical oracle. cross_entropy is float32 with mean
reduction over the batch, matching torch.nn.functional.cross_entropy.
Labels are integer class indices stored as float32 Tensors (milestone 1).
"""

import gc
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mojodl

CE_RTOL = 1e-5
CE_ATOL = 1e-6

_ORACLE_SHAPES = [
    (1, 1),
    (4, 3),
    (8, 10),
    (16, 5),
]

_ORACLE_IDS = ["1x1", "4x3", "8x10", "16x5"]


def _seeded_logits_and_labels(n, c, seed=0):
    """Seeded float32 logits (N, C) and int64 labels (N,)."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, c)).astype(np.float32)
    labels = rng.integers(0, c, n).astype(np.int64)
    return logits, labels


@pytest.mark.parametrize("shape", _ORACLE_SHAPES, ids=_ORACLE_IDS)
def test_cross_entropy_forward_matches_torch(shape):
    n, c = shape
    logits_np, labels_np = _seeded_logits_and_labels(n, c)
    logits = mojodl.Tensor(logits_np)
    labels = mojodl.Tensor(labels_np)

    result = mojodl.cross_entropy(logits, labels)
    expected = F.cross_entropy(
        torch.from_numpy(logits_np), torch.from_numpy(labels_np)
    )

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=CE_RTOL, atol=CE_ATOL
    )


def test_cross_entropy_worked_example_uniform_two_class():
    logits = mojodl.Tensor(np.array([[0.0, 0.0]], dtype=np.float32))
    labels = mojodl.Tensor(np.array([1], dtype=np.int64))

    result = mojodl.cross_entropy(logits, labels)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(
        float(result.numpy()), math.log(2.0), rtol=1e-6
    )


def test_cross_entropy_worked_example_three_class():
    logits = mojodl.Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    labels = mojodl.Tensor(np.array([2], dtype=np.int64))

    result = mojodl.cross_entropy(logits, labels)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    expected = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0)) - 3.0
    np.testing.assert_allclose(float(result.numpy()), expected, rtol=1e-6)
    np.testing.assert_allclose(float(result.numpy()), 0.40760596, rtol=1e-6)


def test_cross_entropy_empty_batch_is_nan():
    logits = mojodl.Tensor(np.zeros((0, 3), dtype=np.float32))
    labels = mojodl.Tensor(np.zeros((0,), dtype=np.int64))

    result = mojodl.cross_entropy(logits, labels)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    assert np.isnan(result.numpy())


def test_cross_entropy_does_not_mutate_inputs():
    logits_np, labels_np = _seeded_logits_and_labels(4, 3)
    logits = mojodl.Tensor(logits_np)
    labels = mojodl.Tensor(labels_np)
    logits_before = logits.numpy().copy()
    labels_before = labels.numpy().copy()

    _ = mojodl.cross_entropy(logits, labels)

    np.testing.assert_array_equal(logits.numpy(), logits_before)
    np.testing.assert_array_equal(labels.numpy(), labels_before)


def test_cross_entropy_result_storage_is_accounted():
    logits_np, labels_np = _seeded_logits_and_labels(4, 3)
    logits = mojodl.Tensor(logits_np)
    labels = mojodl.Tensor(labels_np)
    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    result = mojodl.cross_entropy(logits, labels)

    assert mojodl.memory_stats()["current_bytes"] - baseline == 4
    del result
    gc.collect()


def test_cross_entropy_meta_returns_meta_without_allocating():
    logits = mojodl.empty((4, 3), device="meta")
    labels = mojodl.empty((4,), device="meta")
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    result = mojodl.cross_entropy(logits, labels)

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == ()
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
