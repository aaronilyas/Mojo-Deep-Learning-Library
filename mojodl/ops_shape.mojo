"""Shape ops: reshape (zero-copy view) and transpose (contiguous copy).

Reshape: target validation and -1 inference are shared so cpu and Meta cannot
diverge. Never allocates; the result shares the input's storage (or Meta
shape-only identity) and holds a strong reference to the storage owner.

Transpose: dim normalization is shared (`_normalize_dim` from ops_reduce) so
cpu and Meta cannot diverge. Cpu allocates a contiguous row-major result of
the same byte size; Meta returns a Meta tensor with swapped shape and no
allocation. dim0 == dim1 still returns a new tensor (copy).
"""

from std.memory import unsafe_memcpy
from std.python import PythonObject
from tensor_core import Tensor, _row_major_strides, _format_shape, _nbytes_for_shape
from ops import _parse_shape, _alloc_fp32_out, _make_meta_result, _make_cpu_result
from ops_reduce import _normalize_dim


def _numel(shape: List[Int]) -> Int:
    """Product of dims; empty shape () is 1 element (a scalar)."""
    var n = 1
    for i in range(len(shape)):
        n = n * shape[i]
    return n


def _resolve_reshape_shape(
    in_shape: List[Int], target: List[Int]
) raises -> List[Int]:
    """Validate `target` and infer at most one -1; shared by Reference and Meta.

    Raises tagged ShapeError for: more than one -1, a negative dim other than
    -1, element-count mismatch (message names input and target shapes), and
    -1 with a zero-element input when the inferred size is ambiguous (PyTorch
    rule: another dim is 0 so known product is 0).
    """
    var in_numel = _numel(in_shape)
    var infer_at = -1
    var known_prod = 1
    for i in range(len(target)):
        var d = target[i]
        if d == -1:
            if infer_at >= 0:
                raise Error(
                    "mojodl.ShapeError: only one dimension can be inferred"
                )
            infer_at = i
        elif d < 0:
            raise Error(
                "mojodl.ShapeError: invalid shape dimension "
                + String(d)
                + " at index "
                + String(i)
                + " of shape "
                + _format_shape(target)
            )
        else:
            known_prod = known_prod * d

    var out = List[Int](capacity=len(target))
    for i in range(len(target)):
        out.append(target[i])

    if infer_at >= 0:
        if known_prod == 0:
            if in_numel == 0:
                raise Error(
                    "mojodl.ShapeError: cannot reshape tensor of 0 elements into shape "
                    + _format_shape(target)
                    + " because the unspecified dimension size -1 can be any value and is ambiguous"
                )
            raise Error(
                "mojodl.ShapeError: cannot reshape tensor of shape "
                + _format_shape(in_shape)
                + " ("
                + String(in_numel)
                + " elements) into shape "
                + _format_shape(target)
            )
        if in_numel % known_prod != 0:
            raise Error(
                "mojodl.ShapeError: cannot reshape tensor of shape "
                + _format_shape(in_shape)
                + " ("
                + String(in_numel)
                + " elements) into shape "
                + _format_shape(target)
            )
        out[infer_at] = in_numel // known_prod
    else:
        var out_numel = _numel(out)
        if out_numel != in_numel:
            raise Error(
                "mojodl.ShapeError: cannot reshape tensor of shape "
                + _format_shape(in_shape)
                + " ("
                + String(in_numel)
                + " elements) into shape "
                + _format_shape(target)
            )

    return out^


def reshape(a_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    """Reshape to `shape_obj`: zero-copy view on cpu and Meta alike.

    `shape_obj` is a sequence of ints (may include one -1). Validation and
    -1 inference come from `_resolve_reshape_shape`. The result shares the
    input's data pointer and nbytes, holds a strong reference to the storage
    owner, and never allocates or frees through accounting.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var target = _parse_shape(shape_obj)
    var out_shape = _resolve_reshape_shape(a[].shape, target)
    var out_strides = _row_major_strides(out_shape)

    # View of a view references the original owner; otherwise this tensor owns.
    # Branch on the Mojo Bool only — never Python truthiness on storage_owner.
    var owner: PythonObject
    if a[].is_view:
        owner = PythonObject(copy=a[].storage_owner)
    else:
        owner = PythonObject(copy=a_obj)

    # Copy the data pointer without taking ownership of the storage.
    var shared: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    if a[].data:
        shared = Optional(a[].data.value())
    else:
        shared = None

    var view = Tensor(
        shared^,
        a[].nbytes,
        out_shape^,
        out_strides^,
        a[].dtype_code,
        a[].device_code,
        True,  # is_view
        owner^,
    )
    return PythonObject(alloc=view^)


def _swapped_shape(shape: List[Int], d0: Int, d1: Int) -> List[Int]:
    """Copy `shape` with axes `d0` and `d1` exchanged (no-op when equal)."""
    var out = List[Int](capacity=len(shape))
    for i in range(len(shape)):
        out.append(shape[i])
    if d0 != d1:
        var tmp = out[d0]
        out[d0] = out[d1]
        out[d1] = tmp
    return out^


def _transpose_in_offset(
    linear: Int,
    out_strides: List[Int],
    in_strides: List[Int],
    d0: Int,
    d1: Int,
) -> Int:
    """Map a row-major linear index in the transposed output to the input offset.

    Out axis `d0` carries the data that lived on in axis `d1` (and vice versa).
    """
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
        var in_axis = d
        if d == d0:
            in_axis = d1
        elif d == d1:
            in_axis = d0
        off += coord * in_strides[in_axis]
    return off


def transpose(
    a_obj: PythonObject, dim0_obj: PythonObject, dim1_obj: PythonObject
) raises -> PythonObject:
    """Swap two axes: contiguous copy on cpu, shape-only on Meta.

    Dims are normalized by `_normalize_dim` once up front (shared by Reference
    and Meta). Cpu allocates exactly the input byte size through accounting
    and writes a contiguous row-major result. dim0 == dim1 still returns a new
    tensor (memcpy copy). Meta returns a Meta tensor with swapped shape and
    no allocation.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var dim0 = Int(py=dim0_obj)
    var dim1 = Int(py=dim1_obj)

    # Validate dims before any allocation (Meta and cpu share one check).
    var rank = len(a[].shape)
    var d0 = _normalize_dim(dim0, rank)
    var d1 = _normalize_dim(dim1, rank)
    var out_shape = _swapped_shape(a[].shape, d0, d1)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if a[].device_code == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        if d0 == d1:
            unsafe_memcpy(
                dest=data.value(), src=a[].data.value(), count=nbytes
            )
        else:
            var a_f = a[].data.value().unsafe_bitcast[Float32]()
            var o_f = data.value().unsafe_bitcast[Float32]()
            var n = nbytes // 4
            for i in range(n):
                var in_idx = _transpose_in_offset(
                    i, out_strides, a[].strides, d0, d1
                )
                o_f[unsafe_offset=i] = a_f[unsafe_offset=in_idx]

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
