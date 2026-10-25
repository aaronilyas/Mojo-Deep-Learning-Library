"""Reference kernels for tensor operations.

Plain, readable CPU implementations used as ground truth. Shape checks and
allocation happen here so Python only remaps tagged Mojo errors.
"""

from std.memory import Pointer
from std.python import PythonObject
from accounting import accounted_alloc
from tensor_core import Tensor, _row_major_strides


def _format_shape(shape: List[Int]) -> String:
    """Format a shape as a Python-tuple-like string, e.g. "(2, 3)" or "(7,)"."""
    var s = String("(")
    for i in range(len(shape)):
        if i > 0:
            s += ", "
        s += String(shape[i])
    if len(shape) == 1:
        s += ","
    s += ")"
    return s


def _shapes_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _raise_shape_error(a: List[Int], b: List[Int]) raises:
    """Raise a tagged Error that the Python veneer maps to mojodl.ShapeError."""
    raise Error(
        "mojodl.ShapeError: shapes "
        + _format_shape(a)
        + " and "
        + _format_shape(b)
        + " are incompatible"
    )


def add(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """Same-shape elementwise add Reference kernel.

    Allocates the output through the accounting layer. Raises a tagged
    ShapeError when shapes differ. Returns a new engine Tensor.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    if not _shapes_equal(a[].shape, b[].shape):
        _raise_shape_error(a[].shape, b[].shape)

    var nbytes = a[].nbytes
    var out_shape = List[Int](capacity=len(a[].shape))
    for i in range(len(a[].shape)):
        out_shape.append(a[].shape[i])
    var out_strides = _row_major_strides(out_shape)

    var data: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    if nbytes < 1:
        data = None
    else:
        data = Optional(accounted_alloc(nbytes))
        var a_f = a[].data.value().unsafe_bitcast[Float32]()
        var b_f = b[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = a_f[unsafe_offset=i] + b_f[unsafe_offset=i]

    var result = Tensor(
        data^,
        nbytes,
        out_shape^,
        out_strides^,
        0,
    )
    return PythonObject(alloc=result^)
