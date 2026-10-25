"""Tests for overflow rejection on empty() and clean mapped error messages.

Review findings: oversized shapes must raise OverflowError without allocating,
and ShapeError / DeviceError messages must not leak internal Mojo type tags.
"""

import gc

import numpy as np
import pytest

import mojodl

# Shapes whose fp32 byte count overflows a signed 64-bit (or wraps a usize).
_OVERSIZED_SHAPES = [
    (2**62 + 16,),
    (2**31, 2**31, 4),
]


@pytest.mark.parametrize("shape", _OVERSIZED_SHAPES, ids=["wrap_1d", "wrap_3d"])
def test_empty_oversized_shape_raises_overflow_error(shape):
    baseline = mojodl.memory_stats()["current_bytes"]

    with pytest.raises(OverflowError):
        mojodl.empty(shape)

    assert mojodl.memory_stats()["current_bytes"] == baseline


@pytest.mark.parametrize("shape", _OVERSIZED_SHAPES, ids=["wrap_1d", "wrap_3d"])
def test_empty_oversized_shape_on_meta_raises_overflow_error(shape):
    with pytest.raises(OverflowError):
        mojodl.empty(shape, device="meta")


def test_shape_error_message_omits_internal_tag():
    a = mojodl.Tensor(np.zeros((2, 3), dtype=np.float32))
    b = mojodl.Tensor(np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(mojodl.ShapeError) as exc_info:
        a + b

    msg = str(exc_info.value)
    assert "mojodl.ShapeError" not in msg
    assert "(2, 3)" in msg
    assert "(3, 2)" in msg


def test_device_error_message_omits_internal_tag():
    a = mojodl.Tensor(np.zeros((2, 3), dtype=np.float32))
    b = mojodl.empty((2, 3), device="meta")

    with pytest.raises(mojodl.DeviceError) as exc_info:
        a + b

    msg = str(exc_info.value)
    assert "mojodl.DeviceError" not in msg
    assert "cpu" in msg
    assert "meta" in msg

    del a, b
    gc.collect()
