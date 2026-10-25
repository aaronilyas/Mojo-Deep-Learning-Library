"""mojodl — Python API backed by a compiled Mojo engine."""

from __future__ import annotations

try:
    from . import _engine
except ImportError as e:
    raise ImportError(
        "mojodl engine is not built; run `pixi run build` first"
    ) from e

import numpy as np

memory_stats = _engine.memory_stats
reset_peak_memory_stats = _engine.reset_peak_memory_stats


class ShapeError(ValueError):
    """Raised when tensor shapes are incompatible for an operation."""


class DeviceError(ValueError):
    """Raised when an operation mixes devices or needs data a device lacks."""


class AutogradError(RuntimeError):
    """Raised when autograd cannot proceed (e.g. saved tensors already freed)."""


# Mojo raises Error with a stable type tag in the message; map tags here so
# later ops (budget errors, etc.) share one remapping path. Tag prefixes are
# stripped so public exception messages stay clean.
_MOJO_ERROR_TAGS: tuple[tuple[str, type[Exception]], ...] = (
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


def _call_engine(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        mapped = _map_mojo_error(e)
        if mapped is e:
            raise
        raise mapped from e


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
    if dtype is not float32 and dtype != float32:
        raise TypeError(
            f"empty(): only mojodl.float32 is supported, got {dtype!r}"
        )
    if device == "cpu":
        device_code = _DEVICE_CPU
    elif device == "meta":
        device_code = _DEVICE_META
    else:
        raise ValueError(
            f"empty(): unknown device {device!r}; expected 'cpu' or 'meta'"
        )
    if not isinstance(shape, (tuple, list)):
        raise TypeError(
            f"empty(): shape must be a tuple or list of ints, got {type(shape).__name__}"
        )
    dims = []
    for i, d in enumerate(shape):
        if isinstance(d, bool) or not isinstance(d, int):
            raise TypeError(
                f"empty(): shape[{i}] must be an int, got {type(d).__name__}"
            )
        if d < 0:
            raise ValueError(
                f"empty(): shape[{i}] must be non-negative, got {d}"
            )
        dims.append(d)
    t = Tensor._from_core(_call_engine(_engine.empty, tuple(dims), device_code))
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

    def sum(self):
        """Reduce all elements to a scalar tensor (shape ())."""
        from . import autograd

        result = Tensor._from_core(_call_engine(_engine.sum, self._core))
        return autograd.record_sum(result, self)

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


__all__ = [
    "Tensor",
    "float32",
    "empty",
    "ShapeError",
    "DeviceError",
    "AutogradError",
    "memory_stats",
    "reset_peak_memory_stats",
    "no_grad",
    "is_grad_enabled",
]
