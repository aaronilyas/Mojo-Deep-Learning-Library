"""Constructor Reference kernels and Meta versions (zeros, ones, later rand/randn).

Fill and allocate through the same helpers as other ops. Meta paths allocate
nothing. Public Python constructors validate args then call these.
"""

from std.python import PythonObject
from ops import (
    _alloc_fp32_out,
    _make_meta_result,
    _make_cpu_result,
    _parse_shape,
)
from tensor_core import _row_major_strides, _nbytes_for_shape


def _fill_constant(
    shape_obj: PythonObject, device_code_obj: PythonObject, value: Float32
) raises -> PythonObject:
    """Allocate a cpu tensor filled with `value`, or a Meta tensor (no storage)."""
    var device_code = Int(py=device_code_obj)
    var out_shape = _parse_shape(shape_obj)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if device_code == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = value

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def zeros(shape_obj: PythonObject, device_code_obj: PythonObject) raises -> PythonObject:
    """Create a tensor filled with 0.0 (Reference) or a Meta tensor (no storage).

    Cpu allocates through accounting and fills every element with 0.0; Meta
    allocates nothing.
    """
    return _fill_constant(shape_obj, device_code_obj, Float32(0.0))
