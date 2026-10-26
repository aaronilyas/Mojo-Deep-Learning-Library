"""Optimizers: SGD (with optional momentum), Adam, and shared Optimizer base.

Parameter-list ownership and validation live on ``Optimizer``. Update math
runs in the Mojo engine (``sgd_step`` / ``sgd_momentum_step`` / ``adam_step``)
so Meta dry-runs allocate nothing. Optimizer-state buffers are allocated
through a single helper on ``Optimizer``. ``zero_grad`` is a later slice.
"""

from __future__ import annotations

import math
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


class Adam(Optimizer):
    """Adam optimizer matching ``torch.optim.Adam`` (weight_decay=0, amsgrad=False).

    Per parameter on each update: ``step += 1`` (Python int);
    ``m ← β1·m + (1−β1)·g``; ``v ← β2·v + (1−β2)·g·g``; bias corrections
    ``bc1 = 1 − β1^step``, ``bc2 = 1 − β2^step`` (Python floats);
    ``step_size = lr / bc1``; ``denom = sqrt(v) / sqrt(bc2) + eps``;
    ``p ← p − step_size · m / denom``. Moments are zero-initialized lazily
    via ``_allocate_state`` on a parameter's first update. Parameters whose
    ``.grad`` is None are skipped and get no state. Element math runs in the
    Mojo ``adam_step`` kernel in place; after state exists, ``step`` allocates
    nothing.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        super().__init__(params)
        self.lr = float(lr)
        self.betas = (float(beta1), float(beta2))
        self.eps = float(eps)
        # id(param) -> first / second moment; step count is a Python int
        self._exp_avg: dict[int, Tensor] = {}
        self._exp_avg_sq: dict[int, Tensor] = {}
        self._steps: dict[int, int] = {}

    def step(self) -> None:
        lr = self.lr
        beta1, beta2 = self.betas
        eps = self.eps
        exp_avg = self._exp_avg
        exp_avg_sq = self._exp_avg_sq
        steps = self._steps
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            key = id(p)
            m = exp_avg.get(key)
            if m is None:
                m = self._allocate_state(p)
                v = self._allocate_state(p)
                exp_avg[key] = m
                exp_avg_sq[key] = v
                steps[key] = 0
            else:
                v = exp_avg_sq[key]
            step = steps[key] + 1
            steps[key] = step
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            step_size = lr / bc1
            sqrt_bc2 = math.sqrt(bc2)
            _call_engine(
                _engine.adam_step,
                p._core,
                m._core,
                v._core,
                g._core,
                beta1,
                beta2,
                step_size,
                sqrt_bc2,
                eps,
            )


__all__ = ["Optimizer", "SGD", "Adam"]
