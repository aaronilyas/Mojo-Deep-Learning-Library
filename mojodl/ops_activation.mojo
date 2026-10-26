"""Reference kernels for activation ops (relu, exp, log, softmax, …).

Plain, readable CPU implementations used as ground truth. Shape checks and
allocation happen here so Python only remaps tagged Mojo errors. Every op
ships a Reference kernel and a Meta version that share output-shape logic.
"""

from std.math import exp as math_exp
from std.math import log as math_log
from std.python import PythonObject
from ops import (
    _ensure_same_device,
    _alloc_fp32_out,
    _make_meta_result,
    _make_cpu_result,
    _checked_out_shape,
)
from tensor_core import Tensor, _row_major_strides, _nbytes_for_shape, _format_shape

# device_code values match tensor_core: 0 = cpu, 1 = meta


def _same_out_shape(x: List[Int]) -> List[Int]:
    """Output shape for elementwise activations: same as the input.

    Shared by Reference and Meta paths for relu, exp, log, and similar ops.
    """
    var out_shape = List[Int](capacity=len(x))
    for i in range(len(x)):
        out_shape.append(x[i])
    return out_shape^


def _normalize_dim(dim: Int, shape: List[Int]) raises -> Int:
    """Map a possibly-negative dim into [0, ndim); shared by softmax / log_softmax.

    Valid range is [-ndim, ndim-1] for ndim >= 1, and {0, -1} for a 0-d
    tensor. Raises ShapeError naming the original `dim` and `shape` when out
    of range.
    """
    var ndim = len(shape)
    if ndim == 0:
        if dim == 0 or dim == -1:
            return 0
        raise Error(
            "mojodl.ShapeError: dim "
            + String(dim)
            + " is out of range for shape "
            + _format_shape(shape)
        )
    var d = dim
    if d < 0:
        d = d + ndim
    if d < 0 or d >= ndim:
        raise Error(
            "mojodl.ShapeError: dim "
            + String(dim)
            + " is out of range for shape "
            + _format_shape(shape)
        )
    return d


def _product_range(shape: List[Int], start: Int, end: Int) -> Int:
    """Product of shape[start:end]; empty range is 1."""
    var p = 1
    for i in range(start, end):
        p = p * shape[i]
    return p


def relu(x_obj: PythonObject) raises -> PythonObject:
    """Elementwise ReLU: Reference kernel or Meta version.

    Output shape matches the input (`_same_out_shape`). Reference: out = x if
    x > 0 else 0. Meta returns a Meta tensor with no allocation; cpu allocates
    through accounting (nothing when the output has zero elements).
    """
    var x = x_obj.downcast_value_ptr[Tensor]()
    var x_dev = x[].device_code

    var out_shape = _same_out_shape(x[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if x_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var x_f = x[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            var v = x_f[unsafe_offset=i]
            if v > Float32(0.0):
                o_f[unsafe_offset=i] = v
            else:
                o_f[unsafe_offset=i] = 0.0

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def relu_backward(grad_obj: PythonObject, x_obj: PythonObject) raises -> PythonObject:
    """ReLU backward: grad_in = grad where x > 0 else 0 (Reference or Meta).

    Devices must match (DeviceError naming "relu"). Shapes must match via
    `_checked_out_shape` so Reference and Meta share one check. Exact-zero
    inputs get gradient 0 (matches PyTorch). Meta returns a Meta tensor with
    no allocation; cpu allocates through accounting.
    """
    var grad = grad_obj.downcast_value_ptr[Tensor]()
    var x = x_obj.downcast_value_ptr[Tensor]()

    var grad_dev = grad[].device_code
    _ensure_same_device(grad_dev, x[].device_code, "relu")

    var out_shape = _checked_out_shape(grad[].shape, x[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if grad_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var g_f = grad[].data.value().unsafe_bitcast[Float32]()
        var x_f = x[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            if x_f[unsafe_offset=i] > Float32(0.0):
                o_f[unsafe_offset=i] = g_f[unsafe_offset=i]
            else:
                o_f[unsafe_offset=i] = 0.0

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def exp(x_obj: PythonObject) raises -> PythonObject:
    """Elementwise exp: Reference kernel or Meta version.

    Output shape matches the input (`_same_out_shape`). Reference computes
    e^x in Float64 and rounds once to Float32 (underflow → 0, overflow →
    +inf). Meta returns a Meta tensor with no allocation; cpu allocates
    through accounting (nothing when the output has zero elements).
    """
    var x = x_obj.downcast_value_ptr[Tensor]()
    var x_dev = x[].device_code

    var out_shape = _same_out_shape(x[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if x_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var x_f = x[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            var v64 = Float64(x_f[unsafe_offset=i])
            o_f[unsafe_offset=i] = Float32(math_exp(v64))

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def log(x_obj: PythonObject) raises -> PythonObject:
    """Elementwise natural log: Reference kernel or Meta version.

    Output shape matches the input (`_same_out_shape`). Reference computes
    ln(x) in Float64 and rounds once to Float32 (IEEE: log(0)=-inf,
    log(negative)=nan, log(+inf)=+inf). Meta returns a Meta tensor with no
    allocation; cpu allocates through accounting (nothing when the output
    has zero elements).
    """
    var x = x_obj.downcast_value_ptr[Tensor]()
    var x_dev = x[].device_code

    var out_shape = _same_out_shape(x[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if x_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var x_f = x[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            var v64 = Float64(x_f[unsafe_offset=i])
            o_f[unsafe_offset=i] = Float32(math_log(v64))

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def log_backward(grad_obj: PythonObject, x_obj: PythonObject) raises -> PythonObject:
    """Log backward: grad_in = grad / x (Reference or Meta).

    Devices must match (DeviceError naming "log"). Shapes must match via
    `_checked_out_shape` so Reference and Meta share one check. Meta returns
    a Meta tensor with no allocation; cpu allocates through accounting.
    """
    var grad = grad_obj.downcast_value_ptr[Tensor]()
    var x = x_obj.downcast_value_ptr[Tensor]()

    var grad_dev = grad[].device_code
    _ensure_same_device(grad_dev, x[].device_code, "log")

    var out_shape = _checked_out_shape(grad[].shape, x[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if grad_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var g_f = grad[].data.value().unsafe_bitcast[Float32]()
        var x_f = x[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = g_f[unsafe_offset=i] / x_f[unsafe_offset=i]

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def softmax(x_obj: PythonObject, dim_obj: PythonObject) raises -> PythonObject:
    """Softmax along `dim`: Reference kernel or Meta version.

    Output shape matches the input. `dim` is validated/normalized by
    `_normalize_dim` (shared with the Meta path and reusable by log_softmax).
    Reference: per (outer, inner) lane, subtract the lane max, accumulate
    sum of exp in Float64, then write Float32(exp(x - max) / sum). Meta
    returns a Meta tensor with no allocation; cpu allocates through
    accounting (nothing when the output has zero elements).
    """
    var x = x_obj.downcast_value_ptr[Tensor]()
    var x_dev = x[].device_code
    var dim = Int(py=dim_obj)

    # Validate dim before any allocation (Meta and cpu share one check).
    var d = _normalize_dim(dim, x[].shape)
    var ndim = len(x[].shape)
    var outer = 1
    var axis_len = 1
    var inner = 1
    if ndim > 0:
        outer = _product_range(x[].shape, 0, d)
        axis_len = x[].shape[d]
        inner = _product_range(x[].shape, d + 1, ndim)

    var out_shape = _same_out_shape(x[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if x_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var x_f = x[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        for o in range(outer):
            for i in range(inner):
                var base = o * axis_len * inner + i
                var lane_max = Float64(x_f[unsafe_offset=base])
                for r in range(1, axis_len):
                    var v = Float64(x_f[unsafe_offset=base + r * inner])
                    if v > lane_max:
                        lane_max = v
                var lane_sum: Float64 = 0.0
                for r in range(axis_len):
                    var e = math_exp(
                        Float64(x_f[unsafe_offset=base + r * inner]) - lane_max
                    )
                    lane_sum = lane_sum + e
                for r in range(axis_len):
                    var e = math_exp(
                        Float64(x_f[unsafe_offset=base + r * inner]) - lane_max
                    )
                    o_f[unsafe_offset=base + r * inner] = Float32(e / lane_sum)

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def softmax_backward(
    grad_obj: PythonObject, y_obj: PythonObject, dim_obj: PythonObject
) raises -> PythonObject:
    """Softmax backward: grad_in = y * (grad - sum_dim(grad * y)).

    Devices must match (DeviceError naming "softmax"). Shapes must match via
    `_checked_out_shape`. Same dim helper as forward. Per-lane dot product
    of grad and y is accumulated in Float64. Meta returns a Meta tensor with
    no allocation; cpu allocates through accounting.
    """
    var grad = grad_obj.downcast_value_ptr[Tensor]()
    var y = y_obj.downcast_value_ptr[Tensor]()

    var grad_dev = grad[].device_code
    _ensure_same_device(grad_dev, y[].device_code, "softmax")

    var dim = Int(py=dim_obj)
    var out_shape = _checked_out_shape(grad[].shape, y[].shape)
    # Re-validate dim against the shared rank so Meta and cpu agree on errors.
    var d = _normalize_dim(dim, out_shape)
    var ndim = len(out_shape)
    var outer = 1
    var axis_len = 1
    var inner = 1
    if ndim > 0:
        outer = _product_range(out_shape, 0, d)
        axis_len = out_shape[d]
        inner = _product_range(out_shape, d + 1, ndim)

    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if grad_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var g_f = grad[].data.value().unsafe_bitcast[Float32]()
        var y_f = y[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        for o in range(outer):
            for i in range(inner):
                var base = o * axis_len * inner + i
                var dot: Float64 = 0.0
                for r in range(axis_len):
                    var idx = base + r * inner
                    dot = (
                        dot
                        + Float64(g_f[unsafe_offset=idx])
                        * Float64(y_f[unsafe_offset=idx])
                    )
                for r in range(axis_len):
                    var idx = base + r * inner
                    var yi = Float64(y_f[unsafe_offset=idx])
                    var gi = Float64(g_f[unsafe_offset=idx])
                    o_f[unsafe_offset=idx] = Float32(yi * (gi - dot))

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
