"""mojodl — Python API backed by a compiled Mojo engine."""

from __future__ import annotations

try:
    from . import _engine
except ImportError as e:
    raise ImportError(
        "mojodl engine is not built; run `pixi run build` first"
    ) from e

import operator
import os

import numpy as np

memory_stats = _engine.memory_stats
reset_peak_memory_stats = _engine.reset_peak_memory_stats


class ShapeError(ValueError):
    """Raised when tensor shapes are incompatible for an operation."""


class DeviceError(ValueError):
    """Raised when an operation mixes devices or needs data a device lacks."""


class AutogradError(RuntimeError):
    """Raised when autograd cannot proceed (e.g. saved tensors already freed)."""


class AxisError(ShapeError, IndexError):
    """Raised when a reduction dim is out of range for the tensor's rank."""


# Mojo raises Error with a stable type tag in the message; map tags here so
# later ops (budget errors, etc.) share one remapping path. Tag prefixes are
# stripped so public exception messages stay clean.
_MOJO_ERROR_TAGS: tuple[tuple[str, type[Exception]], ...] = (
    ("mojodl.AxisError:", AxisError),
    ("mojodl.ShapeError:", ShapeError),
    ("mojodl.DeviceError:", DeviceError),
    ("OverflowError:", OverflowError),
)


def _map_mojo_error(exc: BaseException) -> BaseException:
    """Translate a tagged Mojo Error into a public exception (tag stripped)."""
    msg = str(exc)
    for tag, cls in _MOJO_ERROR_TAGS:
        if tag in msg:
            idx = msg.find(tag)
            clean = msg[idx + len(tag) :].lstrip()
            return cls(clean)
    return exc


def _require_int_dim(dim, op: str) -> int:
    """Validate ``dim`` for softmax / log_softmax; return a plain Python int.

    Accepts any object with ``__index__`` except ``bool`` (so ``int`` and
    NumPy integer scalars work). Anything else raises ``TypeError`` naming
    ``op`` and the received type.
    """
    if isinstance(dim, bool) or not hasattr(dim, "__index__"):
        raise TypeError(
            f"{op}(): dim must be an integer, got {type(dim).__name__}"
        )
    return operator.index(dim)


def _call_engine(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        mapped = _map_mojo_error(e)
        if mapped is e:
            raise
        raise mapped from e


def _encode_reduce_args(dim, keepdim, method: str) -> tuple[int, int, int]:
    """Validate/encode dim and keepdim for Tensor.sum / Tensor.mean.

    Raises TypeError naming ``method`` when ``dim`` is bool or not
    ``__index__``-able. Returns ``(has_dim, dim_arg, keepdim_arg)`` with
    ``dim_arg`` a plain ``int`` for the engine call.
    """
    if dim is not None:
        dim = _require_int_dim(dim, method)
    has_dim = 0 if dim is None else 1
    dim_arg = 0 if dim is None else dim
    keepdim_arg = 1 if keepdim else 0
    return has_dim, dim_arg, keepdim_arg


class _Float32Dtype:
    """Public dtype tag for contiguous fp32 tensors."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "float32"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Float32Dtype)

    def __hash__(self) -> int:
        return hash("mojodl.float32")


float32 = _Float32Dtype()


_DEVICE_CPU = 0
_DEVICE_META = 1


def _validate_constructor_args(name, shape, dtype, device):
    """Shared validation for empty / zeros / ones. Returns (dims, device_code)."""
    if dtype is not float32 and dtype != float32:
        raise TypeError(
            f"{name}(): only mojodl.float32 is supported, got {dtype!r}"
        )
    if device == "cpu":
        device_code = _DEVICE_CPU
    elif device == "meta":
        device_code = _DEVICE_META
    else:
        raise ValueError(
            f"{name}(): unknown device {device!r}; expected 'cpu' or 'meta'"
        )
    if not isinstance(shape, (tuple, list)):
        raise TypeError(
            f"{name}(): shape must be a tuple or list of ints, got {type(shape).__name__}"
        )
    dims = []
    for i, d in enumerate(shape):
        if isinstance(d, bool) or not isinstance(d, int):
            raise TypeError(
                f"{name}(): shape[{i}] must be an int, got {type(d).__name__}"
            )
        if d < 0:
            raise ValueError(
                f"{name}(): shape[{i}] must be non-negative, got {d}"
            )
        dims.append(d)
    return dims, device_code


def empty(shape, dtype=float32, device="cpu", requires_grad=False):
    """Create an uninitialized cpu tensor or a Meta tensor (shape + dtype only).

    Parameters
    ----------
    shape:
        Tuple or list of non-negative ints.
    dtype:
        Only `mojodl.float32` is accepted.
    device:
        `"cpu"` (default) allocates storage through accounting; `"meta"`
        allocates nothing.
    requires_grad:
        If True, the result is a leaf that tracks ops for `.backward()`
        (PyTorch semantics). Default False.
    """
    dims, device_code = _validate_constructor_args("empty", shape, dtype, device)
    t = Tensor._from_core(_call_engine(_engine.empty, tuple(dims), device_code))
    t._requires_grad = bool(requires_grad)
    return t


def zeros(shape, dtype=float32, device="cpu", requires_grad=False):
    """Create a cpu or Meta tensor filled with 0.0.

    Parameters
    ----------
    shape:
        Tuple or list of non-negative ints.
    dtype:
        Only `mojodl.float32` is accepted.
    device:
        `"cpu"` (default) allocates storage through accounting and fills with
        0.0; `"meta"` allocates nothing.
    requires_grad:
        If True, the result is a leaf that tracks ops for `.backward()`
        (PyTorch semantics). Default False.
    """
    dims, device_code = _validate_constructor_args("zeros", shape, dtype, device)
    t = Tensor._from_core(_call_engine(_engine.zeros, tuple(dims), device_code))
    t._requires_grad = bool(requires_grad)
    return t


def ones(shape, dtype=float32, device="cpu", requires_grad=False):
    """Create a cpu or Meta tensor filled with 1.0.

    Parameters
    ----------
    shape:
        Tuple or list of non-negative ints.
    dtype:
        Only `mojodl.float32` is accepted.
    device:
        `"cpu"` (default) allocates storage through accounting and fills with
        1.0; `"meta"` allocates nothing.
    requires_grad:
        If True, the result is a leaf that tracks ops for `.backward()`
        (PyTorch semantics). Default False.
    """
    dims, device_code = _validate_constructor_args("ones", shape, dtype, device)
    t = Tensor._from_core(_call_engine(_engine.ones, tuple(dims), device_code))
    t._requires_grad = bool(requires_grad)
    return t


def _validate_seed(name, seed):
    """Validate or draw a 64-bit seed for rand / randn. Runs before allocation."""
    if seed is None:
        return int.from_bytes(os.urandom(8), "little")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"{name}(): seed must be an int or None, got {type(seed).__name__}"
        )
    if seed < 0 or seed >= 2**64:
        raise ValueError(
            f"{name}(): seed must be in [0, 2**64), got {seed}"
        )
    return seed


def rand(shape, dtype=float32, device="cpu", requires_grad=False, seed=None):
    """Create a cpu or Meta tensor of fp32 uniforms in [0, 1).

    Parameters
    ----------
    shape:
        Tuple or list of non-negative ints.
    dtype:
        Only `mojodl.float32` is accepted.
    device:
        `"cpu"` (default) allocates storage through accounting and fills with
        uniforms; `"meta"` allocates nothing and does not generate values.
    requires_grad:
        If True, the result is a leaf that tracks ops for `.backward()`
        (PyTorch semantics). Default False.
    seed:
        Optional int in ``[0, 2**64)``. Same seed → bit-identical values.
        ``None`` (default) draws a fresh 64-bit seed from ``os.urandom``.
    """
    dims, device_code = _validate_constructor_args("rand", shape, dtype, device)
    seed_u = _validate_seed("rand", seed)
    t = Tensor._from_core(
        _call_engine(_engine.rand, tuple(dims), device_code, seed_u)
    )
    t._requires_grad = bool(requires_grad)
    return t


def randn(shape, dtype=float32, device="cpu", requires_grad=False, seed=None):
    """Create a cpu or Meta tensor of fp32 standard normals.

    Parameters
    ----------
    shape:
        Tuple or list of non-negative ints.
    dtype:
        Only `mojodl.float32` is accepted.
    device:
        `"cpu"` (default) allocates storage through accounting and fills with
        standard normals; `"meta"` allocates nothing and does not generate
        values.
    requires_grad:
        If True, the result is a leaf that tracks ops for `.backward()`
        (PyTorch semantics). Default False.
    seed:
        Optional int in ``[0, 2**64)``. Same seed → bit-identical values.
        ``None`` (default) draws a fresh 64-bit seed from ``os.urandom``.
    """
    dims, device_code = _validate_constructor_args("randn", shape, dtype, device)
    seed_u = _validate_seed("randn", seed)
    t = Tensor._from_core(
        _call_engine(_engine.randn, tuple(dims), device_code, seed_u)
    )
    t._requires_grad = bool(requires_grad)
    return t


class Tensor:
    """Eager fp32 tensor whose element storage lives in the Mojo engine.

    Construction accepts a nested list or any numeric NumPy array. Values are
    converted to float32 on the Python side, then copied into engine storage.
    `.numpy()` returns a fresh copy; mutating it does not affect the tensor.
    Meta tensors (device `"meta"`) hold shape and dtype only.

    Autograd: pass `requires_grad=True` to track ops. `.backward()` on a scalar
    result fills `.grad` on leaves. Gradients never require grad.
    """

    __slots__ = ("_core", "_requires_grad", "_grad", "_grad_fn")

    def __init__(self, data, requires_grad: bool = False):
        arr = np.asarray(data, dtype=np.float32)
        if arr.dtype == object:
            raise ValueError("cannot create Tensor from ragged or non-numeric data")
        # ascontiguousarray promotes 0-d arrays to shape (1,); only copy when needed.
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._core = _engine.Tensor(arr)
        self._requires_grad = bool(requires_grad)
        self._grad = None
        self._grad_fn = None

    @classmethod
    def _from_core(cls, core) -> Tensor:
        """Wrap an existing engine Tensor without copying storage."""
        obj = object.__new__(cls)
        obj._core = core
        obj._requires_grad = False
        obj._grad = None
        obj._grad_fn = None
        return obj

    @property
    def shape(self) -> tuple:
        return self._core.shape()

    @property
    def dtype(self):
        return float32

    @property
    def device(self) -> str:
        return self._core.device()

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def is_leaf(self) -> bool:
        return self._grad_fn is None

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value) -> None:
        if value is None:
            self._grad = None
            return
        if not isinstance(value, Tensor):
            raise TypeError(
                f"grad must be a Tensor or None, got {type(value).__name__}"
            )
        if value.shape != self.shape:
            raise AutogradError(
                f"attempting to assign a gradient of size '{value.shape}' "
                f"to a tensor of size '{self.shape}'"
            )
        if value.device != self.device:
            raise AutogradError(
                f"attempting to assign a gradient with device type '{value.device}' "
                f"to a tensor with device type '{self.device}'"
            )
        self._grad = value

    @property
    def grad_fn(self):
        return self._grad_fn

    def numpy(self) -> np.ndarray:
        if self.device == "meta":
            raise DeviceError(
                "cannot call .numpy() on a Meta tensor (device 'meta' has no storage)"
            )
        out = np.empty(self.shape, dtype=np.float32)
        self._core.copy_to(out)
        return out

    def __add__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        from . import autograd

        result = Tensor._from_core(
            _call_engine(_engine.broadcast_add, self._core, other._core)
        )
        return autograd.record_add(result, self, other)

    def __radd__(self, other):
        return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        from . import autograd

        result = Tensor._from_core(
            _call_engine(_engine.broadcast_sub, self._core, other._core)
        )
        return autograd.record_sub(result, self, other)

    def __rsub__(self, other):
        return NotImplemented

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        from . import autograd

        result = Tensor._from_core(
            _call_engine(_engine.broadcast_mul, self._core, other._core)
        )
        return autograd.record_mul(result, self, other)

    def __rmul__(self, other):
        return NotImplemented

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        from . import autograd

        result = Tensor._from_core(
            _call_engine(_engine.broadcast_div, self._core, other._core)
        )
        return autograd.record_div(result, self, other)

    def __rtruediv__(self, other):
        return NotImplemented

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        from . import autograd

        result = Tensor._from_core(
            _call_engine(_engine.matmul, self._core, other._core)
        )
        return autograd.record_matmul(result, self, other)

    def __rmatmul__(self, other):
        return NotImplemented

    def sum(self, dim=None, keepdim=False):
        """Reduce all elements, or along one axis, to a tensor.

        ``dim`` is ``None`` (all axes) or a single int (negative counts from
        the end; a 0-d tensor accepts 0 and -1). ``keepdim`` keeps reduced
        axes as size 1. Bool / non-int ``dim`` raises ``TypeError``.
        """
        from . import autograd

        has_dim, dim_arg, keepdim_arg = _encode_reduce_args(dim, keepdim, "sum")
        result = Tensor._from_core(
            _call_engine(
                _engine.sum_dim, self._core, dim_arg, has_dim, keepdim_arg
            )
        )
        return autograd.record_sum(
            result,
            self,
            dim=(dim_arg if has_dim else None),
            keepdim=bool(keepdim),
        )

    def mean(self, dim=None, keepdim=False):
        """Reduce all elements, or along one axis, to their mean.

        ``dim`` is ``None`` (all axes) or a single int (negative counts from
        the end; a 0-d tensor accepts 0 and -1). ``keepdim`` keeps reduced
        axes as size 1. Bool / non-int ``dim`` raises ``TypeError``. An empty
        reduction (zero count) yields NaN.
        """
        from . import autograd

        has_dim, dim_arg, keepdim_arg = _encode_reduce_args(dim, keepdim, "mean")
        result = Tensor._from_core(
            _call_engine(
                _engine.mean_dim, self._core, dim_arg, has_dim, keepdim_arg
            )
        )
        return autograd.record_mean(
            result,
            self,
            dim=(dim_arg if has_dim else None),
            keepdim=bool(keepdim),
        )

    def reshape(self, *shape):
        """Return a zero-copy view with a new shape (one ``-1`` may be inferred).

        Accepts varargs ints or a single tuple/list. Bool or non-int dims raise
        ``TypeError``. Element-count and ``-1`` rules raise ``ShapeError``.
        """
        from . import autograd

        if len(shape) == 0:
            raise TypeError("reshape() needs a shape")
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            dims = shape[0]
        else:
            dims = shape
        parsed = [_require_int_dim(d, "reshape") for d in dims]
        result = Tensor._from_core(
            _call_engine(_engine.reshape, self._core, tuple(parsed))
        )
        return autograd.record_reshape(result, self)

    def relu(self):
        """Elementwise ReLU: max(0, x)."""
        from . import autograd

        result = Tensor._from_core(_call_engine(_engine.relu, self._core))
        return autograd.record_relu(result, self)

    def exp(self):
        """Elementwise exponential: e^x."""
        from . import autograd

        result = Tensor._from_core(_call_engine(_engine.exp, self._core))
        return autograd.record_exp(result, self)

    def log(self):
        """Elementwise natural logarithm: ln(x)."""
        from . import autograd

        result = Tensor._from_core(_call_engine(_engine.log, self._core))
        return autograd.record_log(result, self)

    def softmax(self, dim: int):
        """Softmax along dimension ``dim`` (negative counts from the end)."""
        from . import autograd

        dim = _require_int_dim(dim, "softmax")
        result = Tensor._from_core(
            _call_engine(_engine.softmax, self._core, dim)
        )
        return autograd.record_softmax(result, self, dim)

    def log_softmax(self, dim: int):
        """Log-softmax along dimension ``dim`` (negative counts from the end)."""
        from . import autograd

        dim = _require_int_dim(dim, "log_softmax")
        result = Tensor._from_core(
            _call_engine(_engine.log_softmax, self._core, dim)
        )
        return autograd.record_log_softmax(result, self, dim)

    def backward(self, retain_graph: bool = False) -> None:
        """Compute gradients of this tensor w.r.t. leaves that require grad.

        The tensor must require grad and have exactly one element (any rank);
        otherwise raises ``AutogradError``. When ``retain_graph`` is False
        (default), nodes free tensors saved for backward after the traversal;
        a second backward through those nodes raises ``AutogradError``. Pass
        ``retain_graph=True`` to keep them.
        """
        from . import autograd

        autograd.backward(self, retain_graph=retain_graph)


def no_grad():
    """Disable grad recording for the duration of the ``with`` block."""
    from . import autograd

    return autograd.no_grad()


def is_grad_enabled() -> bool:
    """Return whether ops currently record into the autograd graph."""
    from . import autograd

    return autograd.is_grad_enabled()


def relu(x: Tensor) -> Tensor:
    """Elementwise ReLU: max(0, x). Equivalent to ``x.relu()``."""
    return x.relu()


def exp(x: Tensor) -> Tensor:
    """Elementwise exponential: e^x. Equivalent to ``x.exp()``."""
    return x.exp()


def log(x: Tensor) -> Tensor:
    """Elementwise natural logarithm: ln(x). Equivalent to ``x.log()``."""
    return x.log()


def softmax(x: Tensor, dim: int) -> Tensor:
    """Softmax along dimension ``dim``. Equivalent to ``x.softmax(dim)``."""
    return x.softmax(dim)


def log_softmax(x: Tensor, dim: int) -> Tensor:
    """Log-softmax along dimension ``dim``. Equivalent to ``x.log_softmax(dim)``."""
    return x.log_softmax(dim)


__all__ = [
    "Tensor",
    "float32",
    "empty",
    "zeros",
    "ones",
    "rand",
    "randn",
    "ShapeError",
    "DeviceError",
    "AutogradError",
    "AxisError",
    "memory_stats",
    "reset_peak_memory_stats",
    "no_grad",
    "is_grad_enabled",
    "relu",
    "exp",
    "log",
    "softmax",
    "log_softmax",
]
