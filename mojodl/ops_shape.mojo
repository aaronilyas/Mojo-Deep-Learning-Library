"""Shape ops: reshape as a zero-copy view (Reference and Meta).

Target validation and -1 inference are shared so cpu and Meta cannot diverge.
Reshape never allocates; the result shares the input's storage (or Meta
shape-only identity) and holds a strong reference to the storage owner.
"""

from std.python import PythonObject
from tensor_core import Tensor, _row_major_strides, _format_shape
from ops import _parse_shape


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
