"""Reference kernels for tensor operations.

Plain, readable CPU implementations used as ground truth. Shape checks and
allocation happen here so Python only remaps tagged Mojo errors. Every op
ships a Reference kernel and a Meta version that share output-shape logic.
"""

from std.memory import Pointer
from std.python import PythonObject
from accounting import accounted_alloc
from tensor_core import Tensor, _row_major_strides, _format_shape, _nbytes_for_shape

# device_code values match tensor_core: 0 = cpu, 1 = meta


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


def _device_name(code: Int) -> String:
    if code == 1:
        return "meta"
    return "cpu"


def _raise_device_error(a_code: Int, b_code: Int) raises:
    """Raise a tagged Error that the Python veneer maps to mojodl.DeviceError."""
    raise Error(
        "mojodl.DeviceError: cannot add tensors on devices '"
        + _device_name(a_code)
        + "' and '"
        + _device_name(b_code)
        + "'"
    )


def _checked_out_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    """Shared shape check and output-shape for Reference and Meta add.

    Raises ShapeError when shapes differ; otherwise returns a copy of `a`.
    """
    if not _shapes_equal(a, b):
        _raise_shape_error(a, b)
    var out_shape = List[Int](capacity=len(a))
    for i in range(len(a)):
        out_shape.append(a[i])
    return out_shape^


def add(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """Same-shape elementwise add: Reference kernel or Meta version.

    Devices must match. Output shape comes from `_checked_out_shape` so the
    Reference and Meta paths cannot diverge. Meta returns a Meta tensor with
    no allocation; cpu allocates through accounting.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    var a_dev = a[].device_code
    var b_dev = b[].device_code
    if a_dev != b_dev:
        _raise_device_error(a_dev, b_dev)

    var out_shape = _checked_out_shape(a[].shape, b[].shape)
    var out_strides = _row_major_strides(out_shape)
    # Size from shape (cpu and meta); overflow raises before allocation.
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        var meta_result = Tensor(
            None,
            0,
            out_shape^,
            out_strides^,
            0,
            1,
        )
        return PythonObject(alloc=meta_result^)

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
        0,  # cpu
    )
    return PythonObject(alloc=result^)
