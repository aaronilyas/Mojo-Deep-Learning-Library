"""mojodl — Python API backed by a compiled Mojo engine."""

from __future__ import annotations

try:
    from . import _engine
except ImportError as e:
    raise ImportError(
        "mojodl engine is not built; run `pixi run build` first"
    ) from e

import numpy as np

answer = _engine.answer
memory_stats = _engine.memory_stats
reset_peak_memory_stats = _engine.reset_peak_memory_stats


class ShapeError(ValueError):
    """Raised when tensor shapes are incompatible for an operation."""


# Mojo raises Error with a stable type tag in the message; map tags here so
# later ops (budget errors, etc.) share one remapping path.
_MOJO_ERROR_TAGS: tuple[tuple[str, type[Exception]], ...] = (
    ("mojodl.ShapeError:", ShapeError),
)


def _map_mojo_error(exc: BaseException) -> BaseException:
    """Translate a tagged Mojo Error into a public mojodl exception."""
    msg = str(exc)
    for tag, cls in _MOJO_ERROR_TAGS:
        if tag in msg:
            return cls(msg)
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


class Tensor:
    """Eager fp32 tensor whose element storage lives in the Mojo engine.

    Construction accepts a nested list or any numeric NumPy array. Values are
    converted to float32 on the Python side, then copied into engine storage.
    `.numpy()` returns a fresh copy; mutating it does not affect the tensor.
    """

    __slots__ = ("_core",)

    def __init__(self, data):
        arr = np.asarray(data, dtype=np.float32)
        if arr.dtype == object:
            raise ValueError("cannot create Tensor from ragged or non-numeric data")
        # ascontiguousarray promotes 0-d arrays to shape (1,); only copy when needed.
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._core = _engine.Tensor(arr)

    @classmethod
    def _from_core(cls, core) -> Tensor:
        """Wrap an existing engine Tensor without copying storage."""
        obj = object.__new__(cls)
        obj._core = core
        return obj

    @property
    def shape(self) -> tuple:
        return self._core.shape()

    @property
    def dtype(self):
        return float32

    def numpy(self) -> np.ndarray:
        out = np.empty(self.shape, dtype=np.float32)
        self._core.copy_to(out)
        return out

    def __add__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        return Tensor._from_core(_call_engine(_engine.add, self._core, other._core))

    def __radd__(self, other):
        return NotImplemented


__all__ = [
    "answer",
    "_engine",
    "Tensor",
    "float32",
    "ShapeError",
    "memory_stats",
    "reset_peak_memory_stats",
]
