"""Optimizers: SGD (no momentum) and shared Optimizer base.

Parameter-list ownership and validation live on ``Optimizer``. Update math
runs in the Mojo engine (``sgd_step``) so Meta dry-runs allocate nothing.
Momentum, Adam, and ``zero_grad`` are later slices.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import Tensor, _call_engine, _engine


class Optimizer:
    """Base class for optimizers that update a fixed list of parameters.

    Consumes any iterable of tensors into a list and validates before any
    engine work. Subclasses implement ``step``.
    """

    def __init__(self, params: Iterable[Tensor]) -> None:
        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")
        for i, p in enumerate(param_list):
            if not isinstance(p, Tensor):
                raise TypeError(
                    f"optimizer expected mojodl.Tensor parameters, got "
                    f"{type(p).__name__} at index {i}"
                )
        self.params: list[Tensor] = param_list

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent without momentum.

    ``step`` applies ``p ← p − lr·grad`` in place via the engine kernel for
    every parameter whose ``.grad`` is not None. Momentum other than 0.0 is
    not implemented in this slice.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum != 0.0:
            raise NotImplementedError(
                "SGD momentum is not implemented in this slice; pass momentum=0.0"
            )
        super().__init__(params)
        self.lr = float(lr)
        self.momentum = float(momentum)

    def step(self) -> None:
        lr = self.lr
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            _call_engine(_engine.sgd_step, p._core, g._core, lr)


__all__ = ["Optimizer", "SGD"]
