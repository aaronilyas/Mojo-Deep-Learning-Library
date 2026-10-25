"""Reference kernels for reductions (sum along axes).

Output-shape / dim checks are shared by the Reference and Meta paths.
Backward expands the upstream gradient along reduced dimensions.
"""

from std.python import PythonObject
from tensor_core import Tensor, _row_major_strides, _nbytes_for_shape
from ops import (
    _alloc_fp32_out,
    _make_meta_result,
    _make_cpu_result,
    _parse_shape,
    _shapes_equal,
    _raise_shape_error,
)


def _raise_axis_error(dim: Int, rank: Int) raises:
    """Raise a tagged Error that the Python veneer maps to mojodl.AxisError."""
    raise Error(
        "mojodl.AxisError: dim "
        + String(dim)
        + " is out of range for a tensor of rank "
        + String(rank)
    )


def _normalize_dim(dim: Int, rank: Int) raises -> Int:
    """Map a possibly-negative dim into [0, rank) (rank-0 accepts 0 and -1).

    Raises AxisError naming the original `dim` and `rank` when out of range.
    """
    if rank == 0:
        if dim == 0 or dim == -1:
            return 0
        _raise_axis_error(dim, rank)
    var d = dim
    if d < 0:
        d = d + rank
    if d < 0 or d >= rank:
        _raise_axis_error(dim, rank)
    return d


def _sum_out_shape(
    shape: List[Int], has_dim: Bool, dim: Int, keepdim: Bool
) raises -> List[Int]:
    """Shared dim check and output shape for sum Reference and Meta paths.

    `has_dim` false means reduce over every axis (dim=None). Otherwise `dim`
    is normalized (negative from the end; 0-d accepts 0 and -1). keepdim keeps
    reduced axes as size 1; without it they are removed. dim=None + keepdim
    yields a scalar () or an all-ones shape of the same rank.
    """
    var rank = len(shape)
    if not has_dim:
        if keepdim:
            var out = List[Int](capacity=rank)
            for _ in range(rank):
                out.append(1)
            return out^
        return List[Int]()

    var d = _normalize_dim(dim, rank)
    # 0-d: reducing the sole axis yields a scalar () for both keepdim values.
    if rank == 0:
        return List[Int]()

    if keepdim:
        var out_k = List[Int](capacity=rank)
        for i in range(rank):
            if i == d:
                out_k.append(1)
            else:
                out_k.append(shape[i])
        return out_k^

    var out_r = List[Int](capacity=rank - 1)
    for i in range(rank):
        if i != d:
            out_r.append(shape[i])
    return out_r^


def _product_range(shape: List[Int], start: Int, end: Int) -> Int:
    """Product of shape[start:end]; empty range is 1."""
    var p = 1
    for i in range(start, end):
        p = p * shape[i]
    return p


def sum_dim(
    a_obj: PythonObject,
    dim_obj: PythonObject,
    has_dim_obj: PythonObject,
    keepdim_obj: PythonObject,
) raises -> PythonObject:
    """Sum over all elements or one axis: Reference kernel or Meta version.

    `has_dim_obj` is 0 for dim=None, else 1 with `dim_obj` the axis (may be
    negative). Output shape comes from `_sum_out_shape`. Cpu accumulates in
    Float64 per output element; empty reductions write 0.0. Meta returns a
    Meta tensor of the output shape with no allocation.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var a_dev = a[].device_code
    var has_dim = Int(py=has_dim_obj) != 0
    var dim = Int(py=dim_obj)
    var keepdim = Int(py=keepdim_obj) != 0

    var out_shape = _sum_out_shape(a[].shape, has_dim, dim, keepdim)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var o_f = data.value().unsafe_bitcast[Float32]()
        if a[].nbytes < 1:
            # Empty input / empty reduction axis: every output element is 0.0.
            var n_out = nbytes // 4
            for i in range(n_out):
                o_f[unsafe_offset=i] = Float32(0.0)
        else:
            var a_f = a[].data.value().unsafe_bitcast[Float32]()
            var rank = len(a[].shape)
            # Whole-tensor or 0-d (single element): one Float64 accumulator.
            if (not has_dim) or rank == 0:
                var acc: Float64 = 0.0
                var n = a[].nbytes // 4
                for i in range(n):
                    acc = acc + Float64(a_f[unsafe_offset=i])
                o_f[unsafe_offset=0] = Float32(acc)
            else:
                var d = _normalize_dim(dim, rank)
                var outer = _product_range(a[].shape, 0, d)
                var reduced = a[].shape[d]
                var inner = _product_range(a[].shape, d + 1, rank)
                for o in range(outer):
                    for i in range(inner):
                        var acc: Float64 = 0.0
                        for r in range(reduced):
                            var in_idx = (o * reduced + r) * inner + i
                            acc = acc + Float64(a_f[unsafe_offset=in_idx])
                        var out_idx = o * inner + i
                        o_f[unsafe_offset=out_idx] = Float32(acc)

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def sum_backward(
    grad_obj: PythonObject,
    input_shape_obj: PythonObject,
    dim_obj: PythonObject,
    has_dim_obj: PythonObject,
    keepdim_obj: PythonObject,
) raises -> PythonObject:
    """Broadcast sum's upstream gradient back to the input shape.

    Shared by whole-tensor and one-axis sum backward (including keepdim and
    dim=None). Meta returns a Meta tensor of `input_shape` with no allocation;
    cpu allocates through accounting and copies each grad value along the
    reduced axis (or all axes when has_dim is false). Grad shape must equal
    `_sum_out_shape` or a ShapeError is raised before any allocation.
    """
    var g = grad_obj.downcast_value_ptr[Tensor]()
    var device_code = g[].device_code
    var has_dim = Int(py=has_dim_obj) != 0
    var dim = Int(py=dim_obj)
    var keepdim = Int(py=keepdim_obj) != 0

    var out_shape = _parse_shape(input_shape_obj)
    # Re-validate dim against the input rank so Meta and cpu agree on errors.
    var expected = _sum_out_shape(out_shape, has_dim, dim, keepdim)
    if not _shapes_equal(g[].shape, expected):
        _raise_shape_error(g[].shape, expected)

    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if device_code == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n_out = nbytes // 4
        if g[].nbytes < 1:
            for i in range(n_out):
                o_f[unsafe_offset=i] = Float32(0.0)
        else:
            var g_f = g[].data.value().unsafe_bitcast[Float32]()
            var rank = len(out_shape)
            # Whole-tensor or 0-d: broadcast the single grad value everywhere.
            if (not has_dim) or rank == 0:
                var g_val = g_f[unsafe_offset=0]
                for i in range(n_out):
                    o_f[unsafe_offset=i] = g_val
            else:
                # Contiguous row-major: keepdim and non-keepdim grads share
                # layout, so grad index = o * inner + i (mirrors sum_dim).
                var d = _normalize_dim(dim, rank)
                var outer = _product_range(out_shape, 0, d)
                var reduced = out_shape[d]
                var inner = _product_range(out_shape, d + 1, rank)
                for o in range(outer):
                    for i in range(inner):
                        var g_val = g_f[unsafe_offset=o * inner + i]
                        for r in range(reduced):
                            var out_idx = (o * reduced + r) * inner + i
                            o_f[unsafe_offset=out_idx] = g_val

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
