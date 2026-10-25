"""Reference kernels for linear-algebra ops (2-D matmul and transpose variants).

Plain, readable CPU implementations used as ground truth. Shape checks and
allocation happen here so Python only remaps tagged Mojo errors. Every op
ships a Reference kernel and a Meta version that share output-shape logic.
"""

from std.python import PythonObject
from ops import (
    _ensure_same_device,
    _alloc_fp32_out,
    _make_meta_result,
    _make_cpu_result,
    _raise_shape_error,
)
from tensor_core import Tensor, _row_major_strides, _nbytes_for_shape

# device_code values match tensor_core: 0 = cpu, 1 = meta


def _matmul_out_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    """Shared shape check and output shape for matmul: (M,K) @ (K,N) → (M,N).

    Raises ShapeError when either operand is not 2-D or the inner dims differ.
    """
    if len(a) != 2 or len(b) != 2:
        _raise_shape_error(a, b)
    if a[1] != b[0]:
        _raise_shape_error(a, b)
    var out_shape = List[Int](capacity=2)
    out_shape.append(a[0])
    out_shape.append(b[1])
    return out_shape^


def _matmul_nt_out_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    """Shared shape check for matmul_nt (a @ b.T): (M,K) @ (N,K)^T → (M,N).

    Raises ShapeError when either operand is not 2-D or a's K differs from b's K.
    """
    if len(a) != 2 or len(b) != 2:
        _raise_shape_error(a, b)
    if a[1] != b[1]:
        _raise_shape_error(a, b)
    var out_shape = List[Int](capacity=2)
    out_shape.append(a[0])
    out_shape.append(b[0])
    return out_shape^


def _matmul_tn_out_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    """Shared shape check for matmul_tn (a.T @ b): (K,M)^T @ (K,N) → (M,N).

    Raises ShapeError when either operand is not 2-D or a's K differs from b's K.
    """
    if len(a) != 2 or len(b) != 2:
        _raise_shape_error(a, b)
    if a[0] != b[0]:
        _raise_shape_error(a, b)
    var out_shape = List[Int](capacity=2)
    out_shape.append(a[1])
    out_shape.append(b[1])
    return out_shape^


def matmul(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """2-D matmul (M,K) @ (K,N) → (M,N): Reference kernel or Meta version.

    Devices must match. Output shape comes from `_matmul_out_shape` so the
    Reference and Meta paths cannot diverge. Inner products accumulate in
    Float64 and round to fp32 once per output element. K == 0 yields zeros.
    Meta returns a Meta tensor with no allocation; cpu allocates through
    accounting (nothing when the output has zero elements).
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    var a_dev = a[].device_code
    _ensure_same_device(a_dev, b[].device_code, "matmul")

    var out_shape = _matmul_out_shape(a[].shape, b[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var M = a[].shape[0]
        var K = a[].shape[1]
        var N = b[].shape[1]
        var o_s0 = out_strides[0]
        var o_s1 = out_strides[1]
        var o_f = data.value().unsafe_bitcast[Float32]()
        # K == 0: empty inner product is zero; inputs may have no storage.
        if K == 0:
            for i in range(M):
                for j in range(N):
                    o_f[unsafe_offset = i * o_s0 + j * o_s1] = 0.0
        else:
            var a_s0 = a[].strides[0]
            var a_s1 = a[].strides[1]
            var b_s0 = b[].strides[0]
            var b_s1 = b[].strides[1]
            var a_f = a[].data.value().unsafe_bitcast[Float32]()
            var b_f = b[].data.value().unsafe_bitcast[Float32]()
            for i in range(M):
                for j in range(N):
                    var acc: Float64 = 0.0
                    for k in range(K):
                        var av = Float64(a_f[unsafe_offset = i * a_s0 + k * a_s1])
                        var bv = Float64(b_f[unsafe_offset = k * b_s0 + j * b_s1])
                        acc = acc + av * bv
                    o_f[unsafe_offset = i * o_s0 + j * o_s1] = Float32(acc)

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def matmul_nt(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """a @ b.T without materializing the transpose: Reference or Meta version.

    a is (M,K), b is (N,K); result is (M,N). Reads b as if transposed.
    Same Float64 accumulation, device check, and Meta path as matmul.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    var a_dev = a[].device_code
    _ensure_same_device(a_dev, b[].device_code, "matmul")

    var out_shape = _matmul_nt_out_shape(a[].shape, b[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var M = a[].shape[0]
        var K = a[].shape[1]
        var N = b[].shape[0]
        var o_s0 = out_strides[0]
        var o_s1 = out_strides[1]
        var o_f = data.value().unsafe_bitcast[Float32]()
        if K == 0:
            for i in range(M):
                for j in range(N):
                    o_f[unsafe_offset = i * o_s0 + j * o_s1] = 0.0
        else:
            var a_s0 = a[].strides[0]
            var a_s1 = a[].strides[1]
            var b_s0 = b[].strides[0]
            var b_s1 = b[].strides[1]
            var a_f = a[].data.value().unsafe_bitcast[Float32]()
            var b_f = b[].data.value().unsafe_bitcast[Float32]()
            for i in range(M):
                for j in range(N):
                    var acc: Float64 = 0.0
                    for k in range(K):
                        var av = Float64(a_f[unsafe_offset = i * a_s0 + k * a_s1])
                        var bv = Float64(b_f[unsafe_offset = j * b_s0 + k * b_s1])
                        acc = acc + av * bv
                    o_f[unsafe_offset = i * o_s0 + j * o_s1] = Float32(acc)

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def matmul_tn(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """a.T @ b without materializing the transpose: Reference or Meta version.

    a is (K,M), b is (K,N); result is (M,N). Reads a as if transposed.
    Same Float64 accumulation, device check, and Meta path as matmul.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    var a_dev = a[].device_code
    _ensure_same_device(a_dev, b[].device_code, "matmul")

    var out_shape = _matmul_tn_out_shape(a[].shape, b[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var K = a[].shape[0]
        var M = a[].shape[1]
        var N = b[].shape[1]
        var o_s0 = out_strides[0]
        var o_s1 = out_strides[1]
        var o_f = data.value().unsafe_bitcast[Float32]()
        if K == 0:
            for i in range(M):
                for j in range(N):
                    o_f[unsafe_offset = i * o_s0 + j * o_s1] = 0.0
        else:
            var a_s0 = a[].strides[0]
            var a_s1 = a[].strides[1]
            var b_s0 = b[].strides[0]
            var b_s1 = b[].strides[1]
            var a_f = a[].data.value().unsafe_bitcast[Float32]()
            var b_f = b[].data.value().unsafe_bitcast[Float32]()
            for i in range(M):
                for j in range(N):
                    var acc: Float64 = 0.0
                    for k in range(K):
                        var av = Float64(a_f[unsafe_offset = k * a_s0 + i * a_s1])
                        var bv = Float64(b_f[unsafe_offset = k * b_s0 + j * b_s1])
                        acc = acc + av * bv
                    o_f[unsafe_offset = i * o_s0 + j * o_s1] = Float32(acc)

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
