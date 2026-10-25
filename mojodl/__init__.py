"""mojodl — Python API backed by a compiled Mojo engine."""

try:
    from . import _engine
except ImportError as e:
    raise ImportError(
        "mojodl engine is not built; run `pixi run build` first"
    ) from e

answer = _engine.answer

__all__ = ["answer", "_engine"]
