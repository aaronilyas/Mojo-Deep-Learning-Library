"""Mean squared error loss (mean reduction only)."""

from __future__ import annotations

from . import ShapeError, Tensor


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """Return the mean squared error of ``input`` vs ``target`` as a 0-d tensor.

    Computes the mean over all elements of ``(input - target)^2``. Both
    arguments must be ``Tensor`` with equal shapes (no broadcasting). Gradients
    flow to whichever arguments require grad via composition of existing
    autograd-tracked ops.
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f"mse_loss(): input must be a Tensor, got {type(input).__name__}"
        )
    if not isinstance(target, Tensor):
        raise TypeError(
            f"mse_loss(): target must be a Tensor, got {type(target).__name__}"
        )
    if input.shape != target.shape:
        raise ShapeError(
            f"shapes {input.shape} and {target.shape} are incompatible"
        )
    diff = input - target
    return (diff * diff).mean()
