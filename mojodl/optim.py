"""Optimizers: SGD (with optional momentum) and shared Optimizer base.

Parameter-list ownership and validation live on ``Optimizer``. Update math
runs in the Mojo engine (``sgd_step`` / ``sgd_momentum_step``) so Meta
dry-runs allocate nothing. Optimizer-state buffers are allocated through a
single helper on ``Optimizer``. Adam and ``zero_grad`` are later slices.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import Tensor, _call_engine, _engine, zeros


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

    def _allocate_state(self, param: Tensor) -> Tensor:
        """Allocate a zero-filled optimizer-state buffer for ``param``.

        Same shape and device as ``param``. Cpu storage goes through engine
        accounting; Meta allocates nothing. Every optimizer that needs
        per-parameter state must create buffers only through this helper.
        """
        return zeros(param.shape, device=param.device)

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional classical momentum.

    With ``momentum == 0.0``, ``step`` applies ``p ← p − lr·grad`` in place
    via ``sgd_step`` and allocates no buffers. With ``momentum > 0``, each
    parameter's first update lazily creates a momentum buffer (via
    ``_allocate_state``); subsequent updates run ``sgd_momentum_step`` in
    place: ``buf ← m·buf + grad``, ``p ← p − lr·buf``. Parameters whose
    ``.grad`` is None are skipped.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        super().__init__(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        # id(param) -> momentum buffer; only populated when momentum > 0
        self._momentum_buffers: dict[int, Tensor] = {}

    def step(self) -> None:
        lr = self.lr
        momentum = self.momentum
        if momentum == 0.0:
            for p in self.params:
                g = p.grad
                if g is None:
                    continue
                _call_engine(_engine.sgd_step, p._core, g._core, lr)
            return

        buffers = self._momentum_buffers
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            key = id(p)
            buf = buffers.get(key)
            if buf is None:
                buf = self._allocate_state(p)
                buffers[key] = buf
            _call_engine(
                _engine.sgd_momentum_step,
                p._core,
                buf._core,
                g._core,
                lr,
                momentum,
            )


__all__ = ["Optimizer", "SGD"]
