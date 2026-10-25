"""Broadcasting elementwise ops and gradient reduction.

Reference kernels and Meta versions share `broadcast_out_shape`. The binary
loop is one internal helper parametrized by op code so `-`, `/` are thin
wrappers; `broadcast_add`, `broadcast_mul`, and `broadcast_sub` are exposed.
Unary `neg` supports sub backward.
"""

from std.python import PythonObject
from ops import (
    _ensure_same_device,
    _raise_shape_error,
    _alloc_fp32_out,
    _make_meta_result,
    _make_cpu_result,
    _parse_shape,
    _shapes_equal,
)
from tensor_core import Tensor, _row_major_strides, _nbytes_for_shape

# Op codes for the shared broadcasting binary kernel.
# 0 = add; 1 = sub; 2 = mul; later: 3 = div.


def _apply_binop(op: Int, x: Float32, y: Float32) raises -> Float32:
    """Apply a binary elementwise op by code."""
    if op == 0:  # add
        return x + y
    if op == 1:  # sub
        return x - y
    if op == 2:  # mul
        return x * y
    raise Error("internal: unknown elementwise op code")


def broadcast_out_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    """NumPy-style broadcast output shape; shared by Reference and Meta paths.

    Aligns from the right. A missing leading dim is treated as 1. Raises
    ShapeError naming both original shapes when a dim pair is incompatible.
    """
    var na = len(a)
    var nb = len(b)
    var nout = na
    if nb > nout:
        nout = nb
    var out_shape = List[Int](capacity=nout)
    for i in range(nout):
        # Dim index from the right (0 = last).
        var from_right = nout - 1 - i
        var da: Int = 1
        var db: Int = 1
        if from_right < na:
            da = a[na - 1 - from_right]
        if from_right < nb:
            db = b[nb - 1 - from_right]
        if da == db:
            out_shape.append(da)
        elif da == 1:
            out_shape.append(db)
        elif db == 1:
            out_shape.append(da)
        else:
            _raise_shape_error(a, b)
    return out_shape^


def _aligned_broadcast_strides(
    in_shape: List[Int], in_strides: List[Int], out_shape: List[Int]
) -> List[Int]:
    """Strides for indexing `in` while iterating coordinates of `out_shape`.

    Dims that are missing (leading) or size-1 contribute stride 0 so the
    same input element is reused across the broadcast axis.
    """
    var nin = len(in_shape)
    var nout = len(out_shape)
    var result = List[Int](capacity=nout)
    for i in range(nout):
        var from_right = nout - 1 - i
        if from_right < nin:
            var in_dim = in_shape[nin - 1 - from_right]
            if in_dim == 1:
                result.append(0)
            else:
                result.append(in_strides[nin - 1 - from_right])
        else:
            result.append(0)
    return result^


def _linear_to_offset(
    linear: Int, out_strides: List[Int], bcast_strides: List[Int]
) -> Int:
    """Map a row-major linear index in the output to an offset via bcast strides."""
    var rem = linear
    var off = 0
    var ndim = len(out_strides)
    for d in range(ndim):
        var s = out_strides[d]
        var coord: Int
        if s == 0:
            coord = 0
        else:
            coord = rem // s
            rem = rem % s
        off += coord * bcast_strides[d]
    return off


def _broadcast_binop(
    a_obj: PythonObject,
    b_obj: PythonObject,
    op: Int,
    op_name: String,
) raises -> PythonObject:
    """Shared broadcasting binary op: Reference kernel or Meta version.

    Devices must match. Output shape comes from `broadcast_out_shape`. Meta
    allocates nothing; cpu allocates exactly the output byte count through
    accounting and runs the element loop over engine storage.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    var a_dev = a[].device_code
    _ensure_same_device(a_dev, b[].device_code, op_name)

    var out_shape = broadcast_out_shape(a[].shape, b[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var a_bcast = _aligned_broadcast_strides(a[].shape, a[].strides, out_shape)
    var b_bcast = _aligned_broadcast_strides(b[].shape, b[].strides, out_shape)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var a_f = a[].data.value().unsafe_bitcast[Float32]()
        var b_f = b[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        var ndim = len(out_shape)
        if ndim == 0:
            o_f[unsafe_offset=0] = _apply_binop(
                op,
                a_f[unsafe_offset=0],
                b_f[unsafe_offset=0],
            )
        else:
            for i in range(n):
                var a_off = _linear_to_offset(i, out_strides, a_bcast)
                var b_off = _linear_to_offset(i, out_strides, b_bcast)
                o_f[unsafe_offset=i] = _apply_binop(
                    op,
                    a_f[unsafe_offset=a_off],
                    b_f[unsafe_offset=b_off],
                )

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def broadcast_add(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """Broadcasting elementwise add: Reference kernel or Meta version."""
    return _broadcast_binop(a_obj, b_obj, 0, "add")  # op 0 = add


def broadcast_sub(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """Broadcasting elementwise subtract: Reference kernel or Meta version."""
    return _broadcast_binop(a_obj, b_obj, 1, "subtract")  # op 1 = sub


def broadcast_mul(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """Broadcasting elementwise multiply: Reference kernel or Meta version."""
    return _broadcast_binop(a_obj, b_obj, 2, "multiply")  # op 2 = mul


def _neg_out_shape(in_shape: List[Int]) -> List[Int]:
    """Output shape for elementwise negate: same as input (Reference and Meta)."""
    var out = List[Int](capacity=len(in_shape))
    for i in range(len(in_shape)):
        out.append(in_shape[i])
    return out^


def neg(a_obj: PythonObject) raises -> PythonObject:
    """Elementwise negate: Reference kernel or Meta version.

    Output shape matches the input. Meta allocates nothing; cpu allocates
    through accounting. Internal engine helper for sub backward.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var device_code = a[].device_code

    var out_shape = _neg_out_shape(a[].shape)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if device_code == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var a_f = a[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = -a_f[unsafe_offset=i]

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def reduce_to_shape(
    grad_obj: PythonObject, shape_obj: PythonObject
) raises -> PythonObject:
    """Reduce a gradient to `shape_obj` by summing broadcast dimensions.

    Sums over leading dims present in `grad` but not in the target, and over
    dims where the target size is 1. Devices follow `grad_obj`. Meta returns
    a Meta tensor with the target shape and no allocation; cpu allocates
    through accounting. Internal engine helper for broadcasting backward.
    """
    var g = grad_obj.downcast_value_ptr[Tensor]()
    var device_code = g[].device_code
    var target = _parse_shape(shape_obj)

    # Target must broadcast to grad shape (otherwise reduction is undefined).
    var expected = broadcast_out_shape(target, g[].shape)
    if not _shapes_equal(expected, g[].shape):
        _raise_shape_error(g[].shape, target)

    var out_shape = target^
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if device_code == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n_out = nbytes // 4
        # accounted_alloc is uninitialized; zero before accumulating.
        for i in range(n_out):
            o_f[unsafe_offset=i] = 0.0

        if g[].nbytes >= 1:
            var g_f = g[].data.value().unsafe_bitcast[Float32]()
            var n_grad = g[].nbytes // 4
            var g_strides = g[].strides.copy()
            var t_bcast = _aligned_broadcast_strides(
                out_shape, out_strides, g[].shape
            )
            var g_ndim = len(g[].shape)
            if g_ndim == 0:
                # Scalar grad into scalar target (only reachable path here).
                o_f[unsafe_offset=0] = (
                    o_f[unsafe_offset=0] + g_f[unsafe_offset=0]
                )
            else:
                for i in range(n_grad):
                    var t_off = _linear_to_offset(i, g_strides, t_bcast)
                    o_f[unsafe_offset=t_off] = (
                        o_f[unsafe_offset=t_off] + g_f[unsafe_offset=i]
                    )

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
