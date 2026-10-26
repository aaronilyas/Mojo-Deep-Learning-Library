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


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", _ORACLE_SHAPES, ids=_ORACLE_IDS)
def test_cross_entropy_backward_matches_torch(shape):
    n, c = shape
    logits_np, labels_np = _seeded_logits_and_labels(n, c)

    logits = mojodl.Tensor(logits_np, requires_grad=True)
    labels = mojodl.Tensor(labels_np)
    loss = mojodl.cross_entropy(logits, labels)

    torch_logits = torch.tensor(logits_np, requires_grad=True)
    torch_loss = F.cross_entropy(torch_logits, torch.from_numpy(labels_np))

    np.testing.assert_allclose(
        loss.numpy(), torch_loss.detach().numpy(), rtol=CE_RTOL, atol=CE_ATOL
    )

    loss.backward()
    torch_loss.backward()

    assert isinstance(logits.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        logits.grad.numpy(),
        torch_logits.grad.numpy(),
        rtol=CE_RTOL,
        atol=CE_ATOL,
    )


def test_cross_entropy_worked_gradient_uniform_two_class():
    logits = mojodl.Tensor(
        np.array([[0.0, 0.0]], dtype=np.float32), requires_grad=True
    )
    labels = mojodl.Tensor(np.array([1], dtype=np.int64))

    mojodl.cross_entropy(logits, labels).backward()

    assert isinstance(logits.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        logits.grad.numpy(),
        np.array([[0.5, -0.5]], dtype=np.float32),
        atol=1e-7,
    )


def test_cross_entropy_upstream_grad_scales_by_three():
    logits_np, labels_np = _seeded_logits_and_labels(4, 3)

    logits = mojodl.Tensor(logits_np, requires_grad=True)
    labels = mojodl.Tensor(labels_np)
    scale = mojodl.Tensor(np.float32(3.0))
    (mojodl.cross_entropy(logits, labels) * scale).backward()

    torch_logits = torch.tensor(logits_np, requires_grad=True)
    F.cross_entropy(torch_logits, torch.from_numpy(labels_np)).backward()

    assert isinstance(logits.grad, mojodl.Tensor)
    np.testing.assert_allclose(
        logits.grad.numpy(),
        3.0 * torch_logits.grad.numpy(),
        rtol=CE_RTOL,
        atol=CE_ATOL,
    )


def test_cross_entropy_recording_requires_grad_and_no_grad():
    logits_np, labels_np = _seeded_logits_and_labels(4, 3)
    labels = mojodl.Tensor(labels_np)

    logits_rg = mojodl.Tensor(logits_np, requires_grad=True)
    loss_rg = mojodl.cross_entropy(logits_rg, labels)
    assert loss_rg.requires_grad is True
    assert loss_rg.grad_fn is not None

    with mojodl.no_grad():
        loss_ng = mojodl.cross_entropy(logits_rg, labels)
    assert loss_ng.requires_grad is False
    assert loss_ng.grad_fn is None

    logits_plain = mojodl.Tensor(logits_np)
    loss_plain = mojodl.cross_entropy(logits_plain, labels)
    assert loss_plain.requires_grad is False
    assert loss_plain.grad_fn is None


def test_cross_entropy_labels_are_constants_no_grad():
    logits_np, labels_np = _seeded_logits_and_labels(4, 3)
    logits = mojodl.Tensor(logits_np, requires_grad=True)
    labels = mojodl.Tensor(labels_np, requires_grad=False)

    mojodl.cross_entropy(logits, labels).backward()

    assert labels.grad is None


def test_cross_entropy_large_magnitudes_finite_and_matches_torch():
    logits_np, labels_np = _seeded_logits_and_labels(8, 5)
    logits_np = (logits_np * 1e4).astype(np.float32)

    logits = mojodl.Tensor(logits_np, requires_grad=True)
    labels = mojodl.Tensor(labels_np)
    loss = mojodl.cross_entropy(logits, labels)

    torch_logits = torch.tensor(logits_np, requires_grad=True)
    torch_loss = F.cross_entropy(torch_logits, torch.from_numpy(labels_np))

    assert np.isfinite(loss.numpy())
    np.testing.assert_allclose(
        loss.numpy(), torch_loss.detach().numpy(), rtol=CE_RTOL, atol=CE_ATOL
    )

    loss.backward()
    torch_loss.backward()

    assert np.all(np.isfinite(logits.grad.numpy()))
    np.testing.assert_allclose(
        logits.grad.numpy(),
        torch_logits.grad.numpy(),
        atol=CE_ATOL,
    )

    lit_np = np.array(
        [[1e4, -1e4, 0.0], [-1e4, 1e4, 1e4]], dtype=np.float32
    )
    lit_labels_np = np.array([1, 0], dtype=np.int64)

    lit_logits = mojodl.Tensor(lit_np, requires_grad=True)
    lit_labels = mojodl.Tensor(lit_labels_np)
    lit_loss = mojodl.cross_entropy(lit_logits, lit_labels)

    torch_lit = torch.tensor(lit_np, requires_grad=True)
    torch_lit_loss = F.cross_entropy(
        torch_lit, torch.from_numpy(lit_labels_np)
    )

    assert np.isfinite(lit_loss.numpy())
    np.testing.assert_allclose(
        lit_loss.numpy(),
        torch_lit_loss.detach().numpy(),
        rtol=CE_RTOL,
        atol=CE_ATOL,
    )

    lit_loss.backward()
    torch_lit_loss.backward()

    assert np.all(np.isfinite(lit_logits.grad.numpy()))
    np.testing.assert_allclose(
        lit_logits.grad.numpy(),
        torch_lit.grad.numpy(),
        atol=CE_ATOL,
    )


def test_cross_entropy_meta_backward_fills_meta_grad_without_allocating():
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    logits = mojodl.empty((4, 3), device="meta", requires_grad=True)
    labels = mojodl.empty((4,), device="meta")
    mojodl.cross_entropy(logits, labels).backward()

    assert isinstance(logits.grad, mojodl.Tensor)
    assert logits.grad.device == "meta"
    assert logits.grad.shape == (4, 3)

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


# ---------------------------------------------------------------------------
# errors: LabelError, ShapeError, DeviceError, TypeError
# ---------------------------------------------------------------------------


def test_label_error_is_value_error_and_index_error():
    assert issubclass(mojodl.LabelError, ValueError)
    assert issubclass(mojodl.LabelError, IndexError)
    assert "LabelError" in mojodl.__all__


@pytest.mark.parametrize(
    "labels_np,needles",
    [
        (
            np.array([0, 1, 5, 2], dtype=np.float32),
            ("label 5", "3 classes", "index 2"),
        ),
        (
            np.array([0, -1, 1, 2], dtype=np.float32),
            ("label -1", "3 classes"),
        ),
        (
            np.array([3, 0, 0, 0], dtype=np.float32),
            ("label 3",),
        ),
    ],
    ids=["oob_5", "negative", "label_eq_C"],
)
def test_cross_entropy_out_of_range_label_raises_label_error(labels_np, needles):
    logits = mojodl.Tensor(np.zeros((4, 3), dtype=np.float32))
    labels = mojodl.Tensor(labels_np)

    with pytest.raises(mojodl.LabelError) as exc_info:
        mojodl.cross_entropy(logits, labels)

    assert isinstance(exc_info.value, ValueError)
    assert isinstance(exc_info.value, IndexError)
    msg = str(exc_info.value)
    for needle in needles:
        assert needle in msg


def test_cross_entropy_non_integral_label_raises_label_error():
    logits = mojodl.Tensor(np.zeros((4, 3), dtype=np.float32))
    labels = mojodl.Tensor(np.array([0, 1.5, 1, 2], dtype=np.float32))

    with pytest.raises(mojodl.LabelError) as exc_info:
        mojodl.cross_entropy(logits, labels)

    assert "1.5" in str(exc_info.value)


def test_cross_entropy_nan_label_raises_label_error():
    logits = mojodl.Tensor(np.zeros((4, 3), dtype=np.float32))
    labels = mojodl.Tensor(
        np.array([0, float("nan"), 1, 2], dtype=np.float32)
    )

    with pytest.raises(mojodl.LabelError) as exc_info:
        mojodl.cross_entropy(logits, labels)

    assert "nan" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "requires_grad",
    [False, True],
    ids=["plain", "requires_grad"],
)
def test_cross_entropy_bad_labels_do_not_allocate(requires_grad):
    logits = mojodl.Tensor(
        np.zeros((4, 3), dtype=np.float32), requires_grad=requires_grad
    )
    labels = mojodl.Tensor(np.array([0, 1, 5, 2], dtype=np.float32))
    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]

    with pytest.raises(mojodl.LabelError):
        mojodl.cross_entropy(logits, labels)

    gc.collect()
    assert mojodl.memory_stats()["current_bytes"] == baseline


@pytest.mark.parametrize(
    "logits_shape,labels_shape",
    [
        ((3,), (3,)),
        ((2, 3, 4), (2,)),
        ((4, 3), (3,)),
        ((4, 3), (4, 1)),
    ],
    ids=["logits_1d", "logits_3d", "batch_mismatch", "labels_2d"],
)
def test_cross_entropy_bad_shapes_raise_shape_error(logits_shape, labels_shape):
    logits = mojodl.Tensor(np.zeros(logits_shape, dtype=np.float32))
    labels = mojodl.Tensor(np.zeros(labels_shape, dtype=np.float32))

    with pytest.raises(mojodl.ShapeError) as exc_info:
        mojodl.cross_entropy(logits, labels)

    if logits_shape == (4, 3) and labels_shape == (3,):
        msg = str(exc_info.value)
        assert "(4, 3)" in msg
        assert "(3,)" in msg


def test_cross_entropy_meta_shape_mismatch_raises_shape_error():
    logits = mojodl.empty((4, 3), device="meta")
    labels = mojodl.empty((3,), device="meta")

    with pytest.raises(mojodl.ShapeError):
        mojodl.cross_entropy(logits, labels)


@pytest.mark.parametrize(
    "logits_device,labels_device",
    [
        ("cpu", "meta"),
        ("meta", "cpu"),
    ],
    ids=["cpu_logits_meta_labels", "meta_logits_cpu_labels"],
)
def test_cross_entropy_mixed_devices_raise_device_error(
    logits_device, labels_device
):
    if logits_device == "cpu":
        logits = mojodl.Tensor(np.zeros((4, 3), dtype=np.float32))
    else:
        logits = mojodl.empty((4, 3), device="meta")
    if labels_device == "cpu":
        labels = mojodl.Tensor(np.zeros((4,), dtype=np.float32))
    else:
        labels = mojodl.empty((4,), device="meta")

    with pytest.raises(mojodl.DeviceError):
        mojodl.cross_entropy(logits, labels)

    del logits, labels
    gc.collect()


def test_cross_entropy_labels_list_raises_type_error():
    logits = mojodl.Tensor(np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(TypeError) as exc_info:
        mojodl.cross_entropy(logits, [0, 1, 2, 1])

    assert "cross_entropy" in str(exc_info.value)


def test_cross_entropy_labels_numpy_raises_type_error():
    logits = mojodl.Tensor(np.zeros((4, 3), dtype=np.float32))
    labels = np.array([0, 1, 2, 1], dtype=np.float32)

    with pytest.raises(TypeError) as exc_info:
        mojodl.cross_entropy(logits, labels)

    assert "cross_entropy" in str(exc_info.value)


def test_cross_entropy_logits_numpy_raises_type_error():
    logits = np.zeros((4, 3), dtype=np.float32)
    labels = mojodl.Tensor(np.array([0, 1, 2, 1], dtype=np.float32))

    with pytest.raises(TypeError) as exc_info:
        mojodl.cross_entropy(logits, labels)

    assert "cross_entropy" in str(exc_info.value)


# ---------------------------------------------------------------------------
# non-finite logits: NaN / ±inf must match PyTorch loss and grad
# ---------------------------------------------------------------------------

_NONFINITE_ROW0_CASES = [
    (np.array([1.0, np.nan, 2.0], dtype=np.float32), 0),
    (np.array([1.0, np.inf, 2.0], dtype=np.float32), 0),
    (np.array([1.0, 2.0, np.nan], dtype=np.float32), 2),
    (np.array([1.0, -np.inf, 2.0], dtype=np.float32), 0),
    (np.array([-np.inf, 1.0, 2.0], dtype=np.float32), 0),
]

_NONFINITE_ROW0_IDS = [
    "nan_non_label",
    "pos_inf_non_label",
    "nan_at_label",
    "neg_inf_non_label",
    "neg_inf_at_label",
]


@pytest.mark.parametrize(
    "row0, label0", _NONFINITE_ROW0_CASES, ids=_NONFINITE_ROW0_IDS
)
def test_cross_entropy_nonfinite_logits_match_torch(row0, label0):
    """Non-finite logits propagate through loss and grad exactly as PyTorch."""
    logits_np = np.array(
        [row0, [0.5, 0.1, 0.2]], dtype=np.float32
    )
    labels_np = np.array([label0, 1], dtype=np.int64)

    logits = mojodl.Tensor(logits_np, requires_grad=True)
    labels = mojodl.Tensor(labels_np)
    loss = mojodl.cross_entropy(logits, labels)

    torch_logits = torch.tensor(logits_np, requires_grad=True)
    torch_loss = F.cross_entropy(torch_logits, torch.from_numpy(labels_np))

    np.testing.assert_allclose(
        loss.numpy(),
        torch_loss.detach().numpy(),
        rtol=CE_RTOL,
        atol=CE_ATOL,
        equal_nan=True,
    )

    loss.backward()
    torch_loss.backward()

    np.testing.assert_allclose(
        logits.grad.numpy(),
        torch_logits.grad.numpy(),
        rtol=CE_RTOL,
        atol=CE_ATOL,
        equal_nan=True,
    )
