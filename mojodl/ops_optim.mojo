"""Optimizer update kernels: Reference and Meta versions.

In-place parameter updates so Step / Dry run share one path. Meta performs
device and shape checks only and allocates nothing.
"""

from std.math import sqrt
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


def adam_step(
    param_obj: PythonObject,
    m_obj: PythonObject,
    v_obj: PythonObject,
    grad_obj: PythonObject,
    beta1_obj: PythonObject,
    beta2_obj: PythonObject,
    step_size_obj: PythonObject,
    sqrt_bc2_obj: PythonObject,
    eps_obj: PythonObject,
) raises:
    """In-place Adam: update moments then param -= step_size * m / denom.

    m ← β1·m + (1−β1)·g; v ← β2·v + (1−β2)·g·g;
    denom = sqrt(v) / sqrt_bc2 + eps; param ← param − step_size · m / denom.
    Bias corrections and step_size are Python scalars. Devices and shapes of
    param, m, v, and grad must match. Meta returns after checks without
    touching storage. Does not allocate; writes into existing engine buffers.
    """
    var param = param_obj.downcast_value_ptr[Tensor]()
    var m_t = m_obj.downcast_value_ptr[Tensor]()
    var v_t = v_obj.downcast_value_ptr[Tensor]()
    var grad = grad_obj.downcast_value_ptr[Tensor]()

    var param_dev = param[].device_code
    _ensure_same_device(param_dev, m_t[].device_code, "adam_step")
    _ensure_same_device(param_dev, v_t[].device_code, "adam_step")
    _ensure_same_device(param_dev, grad[].device_code, "adam_step")
    if not _shapes_equal(param[].shape, m_t[].shape):
        _raise_shape_error(param[].shape, m_t[].shape)
    if not _shapes_equal(param[].shape, v_t[].shape):
        _raise_shape_error(param[].shape, v_t[].shape)
    if not _shapes_equal(param[].shape, grad[].shape):
        _raise_shape_error(param[].shape, grad[].shape)

    if param_dev == 1:  # meta: checks only
        return

    var beta1 = Float32(Float64(py=beta1_obj))
    var beta2 = Float32(Float64(py=beta2_obj))
    var step_size = Float32(Float64(py=step_size_obj))
    var sqrt_bc2 = Float32(Float64(py=sqrt_bc2_obj))
    var eps = Float32(Float64(py=eps_obj))
    var one_m_b1 = Float32(1.0) - beta1
    var one_m_b2 = Float32(1.0) - beta2
    if param[].nbytes >= 1:
        var p_f = param[].data.value().unsafe_bitcast[Float32]()
        var m_f = m_t[].data.value().unsafe_bitcast[Float32]()
        var v_f = v_t[].data.value().unsafe_bitcast[Float32]()
        var g_f = grad[].data.value().unsafe_bitcast[Float32]()
        var n = param[].nbytes // 4
        for i in range(n):
            var g = g_f[unsafe_offset=i]
            var new_m = beta1 * m_f[unsafe_offset=i] + one_m_b1 * g
            var new_v = beta2 * v_f[unsafe_offset=i] + one_m_b2 * g * g
            m_f[unsafe_offset=i] = new_m
            v_f[unsafe_offset=i] = new_v
            var denom = Float32(sqrt(Float64(new_v))) / sqrt_bc2 + eps
            p_f[unsafe_offset=i] = p_f[unsafe_offset=i] - step_size * new_m / denom
