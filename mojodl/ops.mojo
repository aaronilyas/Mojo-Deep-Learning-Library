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


def _raise_device_error(a_code: Int, b_code: Int, op: String) raises:
    """Raise a tagged Error that the Python veneer maps to mojodl.DeviceError.

    `op` is the verb in the message (e.g. "add", "multiply") so callers know
    which operation mixed devices.
    """
    raise Error(
        "mojodl.DeviceError: cannot "
        + op
        + " tensors on devices '"
        + _device_name(a_code)
        + "' and '"
        + _device_name(b_code)
        + "'"
    )


def _ensure_same_device(a_code: Int, b_code: Int, op: String) raises:
    """Device check shared by same-shape binary ops; raises DeviceError naming `op`."""
    if a_code != b_code:
        _raise_device_error(a_code, b_code, op)


def _checked_out_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    """Shared shape check and output-shape for Reference and Meta same-shape ops.

    Raises ShapeError when shapes differ; otherwise returns a copy of `a`.
    """
    if not _shapes_equal(a, b):
        _raise_shape_error(a, b)
    var out_shape = List[Int](capacity=len(a))
    for i in range(len(a)):
        out_shape.append(a[i])
    return out_shape^


def _alloc_fp32_out(nbytes: Int) raises -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    """Allocate cpu output storage through accounting (or None for empty tensors)."""
    if nbytes < 1:
        return None
    return Optional(accounted_alloc(nbytes))


def _make_meta_result(
    var out_shape: List[Int], var out_strides: List[Int]
) raises -> PythonObject:
    """Build a Meta tensor (no storage) for the Meta path of a same-shape binary op."""
    var meta_result = Tensor(
        None,
        0,
        out_shape^,
        out_strides^,
        0,
        1,
    )
    return PythonObject(alloc=meta_result^)


def _make_cpu_result(
    var data: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    nbytes: Int,
    var out_shape: List[Int],
    var out_strides: List[Int],
) raises -> PythonObject:
    """Wrap allocated cpu storage as a Tensor PythonObject."""
    var result = Tensor(
        data^,
        nbytes,
        out_shape^,
        out_strides^,
        0,
        0,  # cpu
    )
    return PythonObject(alloc=result^)


def add(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """Same-shape elementwise add: Reference kernel or Meta version.

    Devices must match. Output shape comes from `_checked_out_shape` so the
    Reference and Meta paths cannot diverge. Meta returns a Meta tensor with
    no allocation; cpu allocates through accounting.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    var a_dev = a[].device_code
    _ensure_same_device(a_dev, b[].device_code, "add")

    var out_shape = _checked_out_shape(a[].shape, b[].shape)
    var out_strides = _row_major_strides(out_shape)
    # Size from shape (cpu and meta); overflow raises before allocation.
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var a_f = a[].data.value().unsafe_bitcast[Float32]()
        var b_f = b[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = a_f[unsafe_offset=i] + b_f[unsafe_offset=i]

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def _whole_sum_out_shape() -> List[Int]:
    """Output shape for whole-tensor sum: always a scalar (shape ()).

    Shared by the Reference kernel and Meta version so the two paths cannot
    diverge. A later axis-reduction ticket can replace or extend this.
    """
    return List[Int]()


def sum(a_obj: PythonObject) raises -> PythonObject:
    """Whole-tensor sum: Reference kernel or Meta version.

    Reduces every element to a scalar (shape ()). Empty input sums to 0.0.
    Meta returns a Meta scalar with no allocation; cpu allocates 4 bytes
    through accounting. Output shape comes from `_whole_sum_out_shape`.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var a_dev = a[].device_code

    var out_shape = _whole_sum_out_shape()
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    var o_f = data.value().unsafe_bitcast[Float32]()
    var acc: Float32 = 0.0
    if a[].nbytes >= 1:
        var a_f = a[].data.value().unsafe_bitcast[Float32]()
        var n = a[].nbytes // 4
        for i in range(n):
            acc = acc + a_f[unsafe_offset=i]
    o_f[unsafe_offset=0] = acc

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def mul(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    """Same-shape elementwise multiply: Reference kernel or Meta version.

    Devices must match. Output shape comes from `_checked_out_shape` so the
    Reference and Meta paths cannot diverge. Meta returns a Meta tensor with
    no allocation; cpu allocates through accounting.
    """
    var a = a_obj.downcast_value_ptr[Tensor]()
    var b = b_obj.downcast_value_ptr[Tensor]()

    var a_dev = a[].device_code
    _ensure_same_device(a_dev, b[].device_code, "multiply")

    var out_shape = _checked_out_shape(a[].shape, b[].shape)
    var out_strides = _row_major_strides(out_shape)
    # Size from shape (cpu and meta); overflow raises before allocation.
    var nbytes = _nbytes_for_shape(out_shape)

    if a_dev == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var a_f = a[].data.value().unsafe_bitcast[Float32]()
        var b_f = b[].data.value().unsafe_bitcast[Float32]()
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = a_f[unsafe_offset=i] * b_f[unsafe_offset=i]

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
