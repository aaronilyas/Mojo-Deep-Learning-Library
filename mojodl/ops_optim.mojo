"""Optimizer update kernels: Reference and Meta versions.

In-place parameter updates so Step / Dry run share one path. Meta performs
device and shape checks only and allocates nothing.
"""

from std.python import PythonObject
from ops import (
    _ensure_same_device,
    _raise_shape_error,
    _shapes_equal,
)
from tensor_core import Tensor


def sgd_step(
    param_obj: PythonObject, grad_obj: PythonObject, lr_obj: PythonObject
) raises:
    """In-place SGD: param -= lr * grad (Reference) or checks only (Meta).

    Devices and shapes must match. Meta returns without touching storage.
    Does not allocate; writes into the parameter's existing engine buffer.
    """
    var param = param_obj.downcast_value_ptr[Tensor]()
    var grad = grad_obj.downcast_value_ptr[Tensor]()

    var param_dev = param[].device_code
    _ensure_same_device(param_dev, grad[].device_code, "sgd_step")
    if not _shapes_equal(param[].shape, grad[].shape):
        _raise_shape_error(param[].shape, grad[].shape)

    if param_dev == 1:  # meta: checks only
        return

    var lr = Float32(Float64(py=lr_obj))
    if param[].nbytes >= 1:
        var p_f = param[].data.value().unsafe_bitcast[Float32]()
        var g_f = grad[].data.value().unsafe_bitcast[Float32]()
        var n = param[].nbytes // 4
        for i in range(n):
            p_f[unsafe_offset=i] = p_f[unsafe_offset=i] - lr * g_f[unsafe_offset=i]


def sgd_momentum_step(
    param_obj: PythonObject,
    buf_obj: PythonObject,
    grad_obj: PythonObject,
    lr_obj: PythonObject,
    momentum_obj: PythonObject,
) raises:
    """In-place SGD with momentum: buf = m*buf + grad; param -= lr*buf.

    Devices and shapes of param, buf, and grad must match. Meta returns after
    checks without touching storage. Does not allocate; writes into the
    existing engine buffers for param and buf. A zero-initialized buf yields
    buf = grad on the first call (classical momentum, dampening=0).
    """
    var param = param_obj.downcast_value_ptr[Tensor]()
    var buf = buf_obj.downcast_value_ptr[Tensor]()
    var grad = grad_obj.downcast_value_ptr[Tensor]()

    var param_dev = param[].device_code
    _ensure_same_device(param_dev, buf[].device_code, "sgd_momentum_step")
    _ensure_same_device(param_dev, grad[].device_code, "sgd_momentum_step")
    if not _shapes_equal(param[].shape, buf[].shape):
        _raise_shape_error(param[].shape, buf[].shape)
    if not _shapes_equal(param[].shape, grad[].shape):
        _raise_shape_error(param[].shape, grad[].shape)

    if param_dev == 1:  # meta: checks only
        return

    var lr = Float32(Float64(py=lr_obj))
    var m = Float32(Float64(py=momentum_obj))
    if param[].nbytes >= 1:
        var p_f = param[].data.value().unsafe_bitcast[Float32]()
        var b_f = buf[].data.value().unsafe_bitcast[Float32]()
        var g_f = grad[].data.value().unsafe_bitcast[Float32]()
        var n = param[].nbytes // 4
        for i in range(n):
            var new_buf = m * b_f[unsafe_offset=i] + g_f[unsafe_offset=i]
            b_f[unsafe_offset=i] = new_buf
            p_f[unsafe_offset=i] = p_f[unsafe_offset=i] - lr * new_buf
