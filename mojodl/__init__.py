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
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._core = _engine.Tensor(arr)

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


__all__ = ["answer", "_engine", "Tensor", "float32"]
