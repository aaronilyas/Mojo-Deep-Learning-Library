"""Constructor Reference kernels and Meta versions (zeros, rand, randn).

Fill and allocate through the same helpers as other ops. Meta paths allocate
nothing. Public Python constructors validate args then call these.

PRNG: SplitMix64 (Steele, Guy; Vigna, Sebastiano). State is local to each
call; the output stream depends only on the seed. No global mutable RNG.
"""

from std.math import log, sqrt, cos, pi
from std.python import PythonObject
from ops import (
    _alloc_fp32_out,
    _make_meta_result,
    _make_cpu_result,
    _parse_shape,
)
from tensor_core import _row_major_strides, _nbytes_for_shape


def _splitmix64_next(mut state: UInt64) -> UInt64:
    """Advance SplitMix64; return the next 64-bit value.

    Algorithm: Steele & Vigna SplitMix64. Same seed always yields the same
    sequence on every platform; there is no process-wide RNG state.
    """
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _uniform01(mut state: UInt64) -> Float32:
    """Next fp32 uniform in [0, 1) from SplitMix64 (top 24 bits / 2^24)."""
    var bits = _splitmix64_next(state) >> 40  # 24 random bits
    return Float32(bits) * (Float32(1.0) / Float32(16777216.0))


def _standard_normal(mut state: UInt64) -> Float32:
    """Next standard-normal sample via Box–Muller (always finite).

    Maps U(0,1] via `1 - U[0,1)` so log never sees 0.
    """
    var u1 = Float64(1.0) - Float64(_uniform01(state))  # (0, 1]
    var u2 = Float64(_uniform01(state))  # [0, 1)
    var r = sqrt(-2.0 * log(u1))
    var z = r * cos(2.0 * pi * u2)
    return Float32(z)


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


def rand(
    shape_obj: PythonObject,
    device_code_obj: PythonObject,
    seed_obj: PythonObject,
) raises -> PythonObject:
    """Create a tensor of fp32 uniforms in [0, 1), or a Meta tensor (no storage).

    Cpu allocates through accounting and fills with SplitMix64 uniforms from
    `seed_obj` (Python int in [0, 2^64)). Meta allocates nothing and does not
    draw values.
    """
    var device_code = Int(py=device_code_obj)
    var out_shape = _parse_shape(shape_obj)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if device_code == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var state = UInt64(py=seed_obj)
    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = _uniform01(state)

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)


def randn(
    shape_obj: PythonObject,
    device_code_obj: PythonObject,
    seed_obj: PythonObject,
) raises -> PythonObject:
    """Create a tensor of fp32 standard normals, or a Meta tensor (no storage).

    Cpu allocates through accounting and fills with Box–Muller samples from
    SplitMix64 uniforms seeded by `seed_obj`. Meta allocates nothing and does
    not draw values. Samples are always finite (never log(0)).
    """
    var device_code = Int(py=device_code_obj)
    var out_shape = _parse_shape(shape_obj)
    var out_strides = _row_major_strides(out_shape)
    var nbytes = _nbytes_for_shape(out_shape)

    if device_code == 1:  # meta
        return _make_meta_result(out_shape^, out_strides^)

    var state = UInt64(py=seed_obj)
    var data = _alloc_fp32_out(nbytes)
    if nbytes >= 1:
        var o_f = data.value().unsafe_bitcast[Float32]()
        var n = nbytes // 4
        for i in range(n):
            o_f[unsafe_offset=i] = _standard_normal(state)

    return _make_cpu_result(data^, nbytes, out_shape^, out_strides^)
