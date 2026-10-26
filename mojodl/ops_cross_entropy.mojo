"""Reference kernels for mean cross-entropy loss.

Plain, readable CPU implementations used as ground truth. Shape checks and
label validation happen here so Python only remaps tagged Mojo errors.
Reference and Meta paths share one check function.
"""

from std.math import exp as math_exp
from std.math import log as math_log
from std.python import PythonObject
from ops import (
    _ensure_same_device,
    _alloc_fp32_out,
    _make_meta_result,
    _make_cpu_result,
)
from tensor_core import Tensor, _row_major_strides, _nbytes_for_shape, _format_shape

# device_code values match tensor_core: 0 = cpu, 1 = meta


def _cross_entropy_check(
    logits_shape: List[Int],
    labels_shape: List[Int],
    logits_dev: Int,
    labels_dev: Int,
) raises:
    """Shared device/shape validation for cross_entropy (cpu, Meta, later backward).

    Devices must match (DeviceError naming "cross_entropy"). Logits must be
    rank 2 and labels exactly (N,) with N = logits.shape[0]; otherwise a
    tagged ShapeError naming both shapes.
    """
    _ensure_same_device(logits_dev, labels_dev, "cross_entropy")
    if (
        len(logits_shape) != 2
        or len(labels_shape) != 1
        or labels_shape[0] != logits_shape[0]
    ):
        raise Error(
            "mojodl.ShapeError: cross_entropy expects logits of shape (N, C) and labels of shape (N,), got "
            + _format_shape(logits_shape)
            + " and "
            + _format_shape(labels_shape)
        )


def cross_entropy(
    logits_obj: PythonObject, labels_obj: PythonObject
) raises -> PythonObject:
    """Mean cross-entropy: fused Reference kernel or Meta version.

    Logits are (N, C); labels are a same-device Tensor of shape (N,) holding
    integral class indices in fp32 storage. Output is a 0-d fp32 scalar (the
    mean over the batch). Shared checks run before any allocation. On cpu,
    every label is validated before allocation and before indexing. Empty
    batch (N = 0) yields NaN. Meta returns a Meta 0-d tensor with no
    allocation.
    """
    var logits = logits_obj.downcast_value_ptr[Tensor]()
    var labels = labels_obj.downcast_value_ptr[Tensor]()

    var logits_dev = logits[].device_code
    _cross_entropy_check(
        logits[].shape, labels[].shape, logits_dev, labels[].device_code
    )

    var n = logits[].shape[0]
    var c = logits[].shape[1]

    var out_shape = List[Int]()
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if logits_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    # Validate every label before allocation / indexing (memory safety).
    if n >= 1:
        var labels_f = labels[].data.value().unsafe_bitcast[Float32]()
        var c_f = Float32(c)
        for i in range(n):
            var v = labels_f[unsafe_offset=i]
            # NaN is not equal to itself; also reject non-integral / out of range.
            var bad = v != v
            if not bad:
                if v < Float32(0.0) or v >= c_f:
                    bad = True
                else:
                    var yi_check = Int(v)
                    if Float32(yi_check) != v:
                        bad = True
            if bad:
                raise Error(
                    "cross_entropy: label "
                    + String(v)
                    + " at index "
                    + String(i)
                    + " is not an integral class index in [0, "
                    + String(c)
                    + ")"
                )

    var data = _alloc_fp32_out(nbytes)
    var o_f = data.value().unsafe_bitcast[Float32]()

    if n == 0:
        o_f[unsafe_offset=0] = Float32(Float64(0.0) / Float64(0.0))
        return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)

    var x_f = logits[].data.value().unsafe_bitcast[Float32]()
    var y_f = labels[].data.value().unsafe_bitcast[Float32]()
    var total: Float64 = 0.0
    for i in range(n):
        var row = i * c
        var lane_max = Float64(x_f[unsafe_offset=row])
        for j in range(1, c):
            var v = Float64(x_f[unsafe_offset=row + j])
            if v > lane_max:
                lane_max = v
        var lane_sum: Float64 = 0.0
        for j in range(c):
            lane_sum = lane_sum + math_exp(
                Float64(x_f[unsafe_offset=row + j]) - lane_max
            )
        var lse = lane_max + math_log(lane_sum)
        var yi = Int(y_f[unsafe_offset=i])
        var loss_i = lse - Float64(x_f[unsafe_offset=row + yi])
        total = total + loss_i
    o_f[unsafe_offset=0] = Float32(total / Float64(n))

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
