"""Reference kernels for activation ops (relu, exp, log; later softmax, …).

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
from tensor_core import Tensor, _row_major_strides, _nbytes_for_shape

# device_code values match tensor_core: 0 = cpu, 1 = meta


def _same_out_shape(x: List[Int]) -> List[Int]:
    """Output shape for elementwise activations: same as the input.

    Shared by Reference and Meta paths for relu, exp, log, and similar ops.
    """
    var out_shape = List[Int](capacity=len(x))
    for i in range(len(x)):
        out_shape.append(x[i])
    return out_shape^


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
