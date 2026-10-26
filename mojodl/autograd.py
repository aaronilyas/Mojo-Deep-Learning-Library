"""Eager autograd: graph nodes, backward traversal, per-op backward rules.

Recording is eager (ops push nodes as they run). Nodes reference the nodes of
their inputs, not the input tensors, and keep alive only tensors the backward
formula needs (saved tensors). Whole-tensor sum saves none — only shape and
device. Leaves are reached through AccumulateGrad, which holds the leaf.
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Tensor


# device_code values match the engine / Python veneer.
_DEVICE_CPU = 0
_DEVICE_META = 1

# Process-wide grad mode (PyTorch-style). Checked on the shared recording path.
_grad_enabled: bool = True


def is_grad_enabled() -> bool:
    """Return whether ops currently record into the autograd graph."""
    return _grad_enabled


@contextmanager
def no_grad():
    """Disable grad recording for the duration of the ``with`` block.

    Restores the previous mode on exit, including when the block raises.
    Nesting is supported.
    """
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev

_FREED_SAVED_MSG = (
    "Trying to backward through the graph a second time "
    "(or directly access saved tensors after they have already been freed). "
    "Specify retain_graph=True if you need to backward through the graph a second time."
)
_NO_GRAD_BACKWARD_MSG = (
    "element 0 of tensors does not require grad and does not have a grad_fn"
)
_NONSCALAR_BACKWARD_MSG = (
    "grad can be implicitly created only for scalar outputs"
)


def _device_code(device: str) -> int:
    if device == "meta":
        return _DEVICE_META
    return _DEVICE_CPU


def _call_engine(fn, *args):
    """Delegate to the package error-mapping helper (lazy to avoid cycles)."""
    from . import _call_engine as call

    return call(fn, *args)


def _wrap(core) -> Tensor:
    from . import Tensor

    return Tensor._from_core(core)


def _raise_freed_saved() -> None:
    from . import AutogradError

    raise AutogradError(_FREED_SAVED_MSG)


class Node:
    """Base class for backward graph nodes.

    Ops that need activations for their backward formula call
    ``save_for_backward``; ``backward(retain_graph=False)`` then releases
    those references after the traversal. Subclasses that never save get
    release and the double-backward error for free (no-ops / no raise).
    """

    # Subclasses set next_functions to a tuple of upstream Node | None.
    next_functions: tuple

    def save_for_backward(self, *tensors: Tensor | None) -> None:
        """Keep tensors alive for ``apply``; cleared on release unless retained."""
        self._saved_tensors = tensors
        self._saved_released = False

    def saved_tensors(self) -> tuple:
        """Return tensors from ``save_for_backward``, or raise if released."""
        if getattr(self, "_saved_released", False):
            _raise_freed_saved()
        return getattr(self, "_saved_tensors", ())

    def release_saved_tensors(self) -> None:
        """Drop saved-tensor references so storage can be freed by refcounting."""
        if getattr(self, "_saved_tensors", None) is not None:
            self._saved_tensors = None
            self._saved_released = True

    def ensure_saved_available(self) -> None:
        """Raise AutogradError if this node saved tensors that were already freed."""
        if getattr(self, "_saved_released", False):
            _raise_freed_saved()

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        """Consume grad_output; return (next_node, grad_input) pairs to route."""
        raise NotImplementedError


class AccumulateGrad(Node):
    """Leaf node: holds the leaf tensor and accumulates into its `.grad`."""

    __slots__ = ("variable", "next_functions")

    def __init__(self, variable: Tensor):
        self.variable = variable
        self.next_functions = ()

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        leaf = self.variable
        if leaf.grad is None:
            # Owned buffer for this leaf alone; never alias grad_output's storage.
            leaf.grad = _wrap(_call_engine(_engine.clone, grad_output._core))
        else:
            _call_engine(_engine.add_inplace, leaf.grad._core, grad_output._core)
        return []


class SumBackward(Node):
    """Sum or mean over axes: broadcast upstream grad (mean scales by 1/count).

    Saves no tensors — only the input shape, dim, keepdim, device, and which
    engine backward to call. `MeanBackward` is a thin subclass that selects
    `mean_backward`.
    """

    __slots__ = (
        "next_functions",
        "input_shape",
        "input_device",
        "dim",
        "keepdim",
        "_backward_fn",
    )

    def __init__(
        self,
        input_node: Node,
        input_shape: tuple,
        input_device: str,
        dim=None,
        keepdim: bool = False,
        *,
        _backward_fn=None,
    ):
        from . import _engine

        self.next_functions = (input_node,)
        self.input_shape = input_shape
        self.input_device = input_device
        self.dim = dim
        self.keepdim = keepdim
        self._backward_fn = (
            _engine.sum_backward if _backward_fn is None else _backward_fn
        )

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        has_dim = 0 if self.dim is None else 1
        dim_arg = 0 if self.dim is None else self.dim
        keepdim_arg = 1 if self.keepdim else 0
        expanded = _wrap(
            _call_engine(
                self._backward_fn,
                grad_output._core,
                self.input_shape,
                dim_arg,
                has_dim,
                keepdim_arg,
            )
        )
        return [(self.next_functions[0], expanded)]


class MeanBackward(SumBackward):
    """Mean over axes: same as SumBackward with the mean_backward engine."""

    __slots__ = ()

    def __init__(
        self,
        input_node: Node,
        input_shape: tuple,
        input_device: str,
        dim=None,
        keepdim: bool = False,
    ):
        from . import _engine

        super().__init__(
            input_node,
            input_shape,
            input_device,
            dim=dim,
            keepdim=keepdim,
            _backward_fn=_engine.mean_backward,
        )


class ReshapeBackward(Node):
    """Reshape: grad_input is grad_output reshaped to the input shape.

    Saves no tensors — only the input shape. Uses the same engine reshape
    view op, so Meta backward allocates nothing.
    """

    __slots__ = ("next_functions", "input_shape")

    def __init__(self, input_node: Node, input_shape: tuple):
        self.next_functions = (input_node,)
        self.input_shape = input_shape

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        grad_input = _wrap(
            _call_engine(_engine.reshape, grad_output._core, self.input_shape)
        )
        return [(self.next_functions[0], grad_input)]


class TransposeBackward(Node):
    """Transpose: grad_input is grad_output transposed with the same dims.

    Saves no tensors — only dim0 and dim1. Uses the same engine transpose
    op, so Meta backward allocates nothing.
    """

    __slots__ = ("next_functions", "dim0", "dim1")

    def __init__(self, input_node: Node, dim0: int, dim1: int):
        self.next_functions = (input_node,)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        grad_input = _wrap(
            _call_engine(
                _engine.transpose, grad_output._core, self.dim0, self.dim1
            )
        )
        return [(self.next_functions[0], grad_input)]


class AddBackward(Node):
    """Elementwise add: route the incoming gradient to each requiring input.

    When an input was broadcast, the upstream gradient is reduced (summed
    over broadcast dims) to that input's shape. Same-shape inputs receive
    the upstream gradient unchanged (no extra allocation). Saves no tensors
    — only the input shapes.
    """

    __slots__ = ("next_functions", "shape_a", "shape_b")

    def __init__(
        self,
        next_a: Node | None,
        next_b: Node | None,
        shape_a: tuple,
        shape_b: tuple,
    ):
        self.next_functions = (next_a, next_b)
        self.shape_a = shape_a
        self.shape_b = shape_b

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        out: list[tuple[Node, Tensor]] = []
        next_a, next_b = self.next_functions
        if next_a is not None:
            if grad_output.shape == self.shape_a:
                grad_a = grad_output
            else:
                grad_a = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, grad_output._core, self.shape_a
                    )
                )
            out.append((next_a, grad_a))
        if next_b is not None:
            if grad_output.shape == self.shape_b:
                grad_b = grad_output
            else:
                grad_b = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, grad_output._core, self.shape_b
                    )
                )
            out.append((next_b, grad_b))
        return out


class SubBackward(Node):
    """Elementwise sub: grad_a = grad reduced to a's shape;
    grad_b = (-grad) reduced to b's shape.

    Reduce is skipped when shapes already match. Saves no tensors — only
    the input shapes. Negation uses the engine `neg` op (Meta-safe).
    """

    __slots__ = ("next_functions", "shape_a", "shape_b")

    def __init__(
        self,
        next_a: Node | None,
        next_b: Node | None,
        shape_a: tuple,
        shape_b: tuple,
    ):
        self.next_functions = (next_a, next_b)
        self.shape_a = shape_a
        self.shape_b = shape_b

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        out: list[tuple[Node, Tensor]] = []
        next_a, next_b = self.next_functions
        if next_a is not None:
            if grad_output.shape == self.shape_a:
                grad_a = grad_output
            else:
                grad_a = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, grad_output._core, self.shape_a
                    )
                )
            out.append((next_a, grad_a))
        if next_b is not None:
            neg_grad = _wrap(_call_engine(_engine.neg, grad_output._core))
            if neg_grad.shape == self.shape_b:
                grad_b = neg_grad
            else:
                grad_b = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, neg_grad._core, self.shape_b
                    )
                )
            out.append((next_b, grad_b))
        return out


class MulBackward(Node):
    """Broadcasting elementwise mul: grad_a = (grad * b) reduced to a's shape,
    grad_b = (grad * a) reduced to b's shape.

    Saves each other-operand only when that gradient is needed (the input
    tensor itself, not a copy). Reduce is skipped when the product already
    has the input's shape.
    """

    __slots__ = (
        "next_functions",
        "_saved_tensors",
        "_saved_released",
        "shape_a",
        "shape_b",
    )

    def __init__(
        self,
        next_a: Node | None,
        next_b: Node | None,
        a: Tensor,
        b: Tensor,
    ):
        self.next_functions = (next_a, next_b)
        self.shape_a = a.shape
        self.shape_b = b.shape
        # grad_b needs a; grad_a needs b — save only what apply will use.
        self.save_for_backward(
            a if next_b is not None else None,
            b if next_a is not None else None,
        )

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        saved_a, saved_b = self.saved_tensors()
        out: list[tuple[Node, Tensor]] = []
        next_a, next_b = self.next_functions
        if next_a is not None:
            prod = _wrap(
                _call_engine(
                    _engine.broadcast_mul, grad_output._core, saved_b._core
                )
            )
            if prod.shape == self.shape_a:
                grad_a = prod
            else:
                grad_a = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, prod._core, self.shape_a
                    )
                )
            out.append((next_a, grad_a))
        if next_b is not None:
            prod = _wrap(
                _call_engine(
                    _engine.broadcast_mul, grad_output._core, saved_a._core
                )
            )
            if prod.shape == self.shape_b:
                grad_b = prod
            else:
                grad_b = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, prod._core, self.shape_b
                    )
                )
            out.append((next_b, grad_b))
        return out


class DivBackward(Node):
    """Broadcasting elementwise div: grad_a = (grad / b) reduced to a's shape,
    grad_b = (-grad * a / b²) reduced to b's shape.

    Built from broadcast_div, broadcast_mul, neg, and reduce_to_shape.
    Saves b whenever any gradient is needed; a only if b requires grad
    (the input tensors themselves). Reduce is skipped when shapes match.
    """

    __slots__ = (
        "next_functions",
        "_saved_tensors",
        "_saved_released",
        "shape_a",
        "shape_b",
    )

    def __init__(
        self,
        next_a: Node | None,
        next_b: Node | None,
        a: Tensor,
        b: Tensor,
    ):
        self.next_functions = (next_a, next_b)
        self.shape_a = a.shape
        self.shape_b = b.shape
        # grad_a needs b; grad_b needs a and b — save only what apply will use.
        self.save_for_backward(
            a if next_b is not None else None,
            b if next_a is not None or next_b is not None else None,
        )

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        saved_a, saved_b = self.saved_tensors()
        out: list[tuple[Node, Tensor]] = []
        next_a, next_b = self.next_functions
        if next_a is not None:
            quot = _wrap(
                _call_engine(
                    _engine.broadcast_div, grad_output._core, saved_b._core
                )
            )
            if quot.shape == self.shape_a:
                grad_a = quot
            else:
                grad_a = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, quot._core, self.shape_a
                    )
                )
            out.append((next_a, grad_a))
        if next_b is not None:
            neg_grad = _wrap(_call_engine(_engine.neg, grad_output._core))
            neg_grad_a = _wrap(
                _call_engine(
                    _engine.broadcast_mul, neg_grad._core, saved_a._core
                )
            )
            b_sq = _wrap(
                _call_engine(
                    _engine.broadcast_mul, saved_b._core, saved_b._core
                )
            )
            quot = _wrap(
                _call_engine(
                    _engine.broadcast_div, neg_grad_a._core, b_sq._core
                )
            )
            if quot.shape == self.shape_b:
                grad_b = quot
            else:
                grad_b = _wrap(
                    _call_engine(
                        _engine.reduce_to_shape, quot._core, self.shape_b
                    )
                )
            out.append((next_b, grad_b))
        return out


def input_node(tensor: Tensor) -> Node | None:
    """Node to wire as the upstream of an op that used `tensor`.

    Non-grad tensors contribute nothing. Results with a grad_fn reuse that
    node. Leaves get a fresh AccumulateGrad that holds the leaf (the leaf
    itself does not point at the node, so there is no reference cycle).
    When grad mode is off, always returns None so record_* skip node creation.
    """
    if not _grad_enabled or not tensor.requires_grad:
        return None
    if tensor.grad_fn is not None:
        return tensor.grad_fn
    return AccumulateGrad(tensor)


def attach(result: Tensor, grad_fn: Node | None) -> Tensor:
    """Shared recording helper: every Tensor op calls this with its result node.

    When grad mode is off or `grad_fn` is None the result is left as a
    non-grad leaf. Otherwise the result requires grad, is not a leaf, and
    carries `grad_fn`.
    """
    if not _grad_enabled or grad_fn is None:
        return result
    result._requires_grad = True
    result._grad_fn = grad_fn
    return result


def record_reshape(result: Tensor, inp: Tensor) -> Tensor:
    """Attach a ReshapeBackward node when `inp` requires grad."""
    nxt = input_node(inp)
    if nxt is None:
        return result
    return attach(result, ReshapeBackward(nxt, inp.shape))


def record_transpose(
    result: Tensor, inp: Tensor, dim0: int, dim1: int
) -> Tensor:
    """Attach a TransposeBackward node when `inp` requires grad."""
    nxt = input_node(inp)
    if nxt is None:
        return result
    return attach(result, TransposeBackward(nxt, dim0, dim1))


def record_sum(result: Tensor, inp: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    """Attach a SumBackward node when `inp` requires grad."""
    nxt = input_node(inp)
    if nxt is None:
        return result
    return attach(
        result, SumBackward(nxt, inp.shape, inp.device, dim=dim, keepdim=keepdim)
    )


def record_mean(result: Tensor, inp: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    """Attach a MeanBackward node when `inp` requires grad."""
    nxt = input_node(inp)
    if nxt is None:
        return result
    return attach(
        result, MeanBackward(nxt, inp.shape, inp.device, dim=dim, keepdim=keepdim)
    )


def record_add(result: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Attach an AddBackward node when either input requires grad."""
    next_a = input_node(a)
    next_b = input_node(b)
    if next_a is None and next_b is None:
        return result
    return attach(result, AddBackward(next_a, next_b, a.shape, b.shape))


def record_sub(result: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Attach a SubBackward node when either input requires grad."""
    next_a = input_node(a)
    next_b = input_node(b)
    if next_a is None and next_b is None:
        return result
    return attach(result, SubBackward(next_a, next_b, a.shape, b.shape))


def record_mul(result: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Attach a MulBackward node when either input requires grad."""
    next_a = input_node(a)
    next_b = input_node(b)
    if next_a is None and next_b is None:
        return result
    return attach(result, MulBackward(next_a, next_b, a, b))


def record_div(result: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Attach a DivBackward node when either input requires grad."""
    next_a = input_node(a)
    next_b = input_node(b)
    if next_a is None and next_b is None:
        return result
    return attach(result, DivBackward(next_a, next_b, a, b))


class MatmulBackward(Node):
    """Matmul: grad_a = grad @ b^T, grad_b = a^T @ grad.

    Saves each other-operand only when that gradient is needed (the input
    tensor itself, not a copy).
    """

    __slots__ = ("next_functions", "_saved_tensors", "_saved_released")

    def __init__(
        self,
        next_a: Node | None,
        next_b: Node | None,
        a: Tensor,
        b: Tensor,
    ):
        self.next_functions = (next_a, next_b)
        # grad_b needs a; grad_a needs b — save only what apply will use.
        self.save_for_backward(
            a if next_b is not None else None,
            b if next_a is not None else None,
        )

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        saved_a, saved_b = self.saved_tensors()
        out: list[tuple[Node, Tensor]] = []
        next_a, next_b = self.next_functions
        if next_a is not None:
            grad_a = _wrap(
                _call_engine(_engine.matmul_nt, grad_output._core, saved_b._core)
            )
            out.append((next_a, grad_a))
        if next_b is not None:
            grad_b = _wrap(
                _call_engine(_engine.matmul_tn, saved_a._core, grad_output._core)
            )
            out.append((next_b, grad_b))
        return out


def record_matmul(result: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Attach a MatmulBackward node when either input requires grad."""
    next_a = input_node(a)
    next_b = input_node(b)
    if next_a is None and next_b is None:
        return result
    return attach(result, MatmulBackward(next_a, next_b, a, b))


class ReluBackward(Node):
    """ReLU: grad_in = grad where x > 0 else 0.

    Saves only the input x (needed to mask the upstream gradient).
    """

    __slots__ = ("next_functions", "_saved_tensors", "_saved_released")

    def __init__(self, next_x: Node, x: Tensor):
        self.next_functions = (next_x,)
        self.save_for_backward(x)

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        (saved_x,) = self.saved_tensors()
        grad_x = _wrap(
            _call_engine(_engine.relu_backward, grad_output._core, saved_x._core)
        )
        return [(self.next_functions[0], grad_x)]


def record_relu(result: Tensor, inp: Tensor) -> Tensor:
    """Attach a ReluBackward node when `inp` requires grad."""
    nxt = input_node(inp)
    if nxt is None:
        return result
    return attach(result, ReluBackward(nxt, inp))


class ExpBackward(Node):
    """Exp: grad_in = grad * exp(x), using the saved forward output.

    Saves a fresh non-grad Tensor wrapping the result's engine core so the
    result object itself is not held (avoids result → grad_fn → saved → result).
    """

    __slots__ = ("next_functions", "_saved_tensors", "_saved_released")

    def __init__(self, next_x: Node, saved_out: Tensor):
        self.next_functions = (next_x,)
        self.save_for_backward(saved_out)

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        (saved_out,) = self.saved_tensors()
        grad_x = _wrap(
            _call_engine(_engine.mul, grad_output._core, saved_out._core)
        )
        return [(self.next_functions[0], grad_x)]


def record_exp(result: Tensor, inp: Tensor) -> Tensor:
    """Attach an ExpBackward node when `inp` requires grad.

    Saves ``Tensor._from_core(result._core)`` (same storage, no grad_fn) so
    dropping ``result`` can free its storage by refcounting.
    """
    from . import Tensor

    nxt = input_node(inp)
    if nxt is None:
        return result
    saved_out = Tensor._from_core(result._core)
    return attach(result, ExpBackward(nxt, saved_out))


class LogBackward(Node):
    """Log: grad_in = grad / x.

    Saves only the input x (needed to divide the upstream gradient).
    """

    __slots__ = ("next_functions", "_saved_tensors", "_saved_released")

    def __init__(self, next_x: Node, x: Tensor):
        self.next_functions = (next_x,)
        self.save_for_backward(x)

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        (saved_x,) = self.saved_tensors()
        grad_x = _wrap(
            _call_engine(_engine.log_backward, grad_output._core, saved_x._core)
        )
        return [(self.next_functions[0], grad_x)]


def record_log(result: Tensor, inp: Tensor) -> Tensor:
    """Attach a LogBackward node when `inp` requires grad."""
    nxt = input_node(inp)
    if nxt is None:
        return result
    return attach(result, LogBackward(nxt, inp))


class SoftmaxBackward(Node):
    """Softmax: grad_in = y * (grad - sum_dim(grad * y)), using the saved output.

    Saves a fresh non-grad Tensor wrapping the result's engine core so the
    result object itself is not held (avoids result → grad_fn → saved → result),
    plus the dim used in the forward pass.
    """

    __slots__ = ("next_functions", "_saved_tensors", "_saved_released", "dim")

    def __init__(self, next_x: Node, saved_out: Tensor, dim: int):
        self.next_functions = (next_x,)
        self.dim = dim
        self.save_for_backward(saved_out)

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        (saved_out,) = self.saved_tensors()
        grad_x = _wrap(
            _call_engine(
                _engine.softmax_backward,
                grad_output._core,
                saved_out._core,
                self.dim,
            )
        )
        return [(self.next_functions[0], grad_x)]


def record_softmax(result: Tensor, inp: Tensor, dim: int) -> Tensor:
    """Attach a SoftmaxBackward node when `inp` requires grad.

    Saves ``Tensor._from_core(result._core)`` (same storage, no grad_fn) so
    dropping ``result`` can free its storage by refcounting.
    """
    from . import Tensor

    nxt = input_node(inp)
    if nxt is None:
        return result
    saved_out = Tensor._from_core(result._core)
    return attach(result, SoftmaxBackward(nxt, saved_out, dim))


class LogSoftmaxBackward(Node):
    """Log-softmax: grad_in = grad - exp(y) * sum_dim(grad), using the saved output.

    Saves a fresh non-grad Tensor wrapping the result's engine core so the
    result object itself is not held (avoids result → grad_fn → saved → result),
    plus the dim used in the forward pass.
    """

    __slots__ = ("next_functions", "_saved_tensors", "_saved_released", "dim")

    def __init__(self, next_x: Node, saved_out: Tensor, dim: int):
        self.next_functions = (next_x,)
        self.dim = dim
        self.save_for_backward(saved_out)

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        (saved_out,) = self.saved_tensors()
        grad_x = _wrap(
            _call_engine(
                _engine.log_softmax_backward,
                grad_output._core,
                saved_out._core,
                self.dim,
            )
        )
        return [(self.next_functions[0], grad_x)]


def record_log_softmax(result: Tensor, inp: Tensor, dim: int) -> Tensor:
    """Attach a LogSoftmaxBackward node when `inp` requires grad.

    Saves ``Tensor._from_core(result._core)`` (same storage, no grad_fn) so
    dropping ``result`` can free its storage by refcounting.
    """
    from . import Tensor

    nxt = input_node(inp)
    if nxt is None:
        return result
    saved_out = Tensor._from_core(result._core)
    return attach(result, LogSoftmaxBackward(nxt, saved_out, dim))


class CrossEntropyBackward(Node):
    """Mean cross-entropy: grad_logits from upstream scalar and saved inputs.

    Saves non-grad aliases of logits and labels (constants; no node for
    labels). Backward calls the fused engine kernel
    ``cross_entropy_backward(grad, logits, labels)``.
    """

    __slots__ = ("next_functions", "_saved_tensors", "_saved_released")

    def __init__(
        self, next_logits: Node, saved_logits: Tensor, saved_labels: Tensor
    ):
        self.next_functions = (next_logits,)
        self.save_for_backward(saved_logits, saved_labels)

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        saved_logits, saved_labels = self.saved_tensors()
        grad_x = _wrap(
            _call_engine(
                _engine.cross_entropy_backward,
                grad_output._core,
                saved_logits._core,
                saved_labels._core,
            )
        )
        return [(self.next_functions[0], grad_x)]


def record_cross_entropy(
    result: Tensor, logits: Tensor, labels: Tensor
) -> Tensor:
    """Attach a CrossEntropyBackward node when `logits` requires grad.

    Labels are constants: never get a node. Saves
    ``Tensor._from_core`` aliases of logits and labels so the result does not
    hold the caller's Tensor objects.
    """
    from . import Tensor

    nxt = input_node(logits)
    if nxt is None:
        return result
    saved_logits = Tensor._from_core(logits._core)
    saved_labels = Tensor._from_core(labels._core)
    return attach(
        result, CrossEntropyBackward(nxt, saved_logits, saved_labels)
    )


def _ones_like_seed(tensor: Tensor) -> Tensor:
    """Engine-created ones tensor with the output's shape and device (seed grad)."""
    from . import _engine

    return _wrap(
        _call_engine(
            _engine.ones, tensor.shape, _device_code(tensor.device)
        )
    )


def _numel(shape: tuple) -> int:
    """Product of dims; empty shape () is 1 (a scalar)."""
    n = 1
    for d in shape:
        n *= d
    return n


def backward(tensor: Tensor, retain_graph: bool = False) -> None:
    """Run reverse-mode AD from `tensor`.

    The output must require grad and have exactly one element (any rank);
    otherwise raises ``AutogradError`` before allocating a seed or touching
    any leaf ``.grad``.

    Visits nodes in reverse topological order. Each node's incoming gradients
    are fully accumulated before its `apply` runs, so a tensor used more than
    once receives the summed gradient.

    When ``retain_graph`` is False, every reachable node releases tensors it
    saved for backward after the traversal. A later backward that reaches a
    released node raises ``AutogradError`` during graph discovery, before any
    leaf ``.grad`` is modified.
    """
    from . import AutogradError

    if not tensor.requires_grad:
        raise AutogradError(_NO_GRAD_BACKWARD_MSG)
    if _numel(tensor.shape) != 1:
        raise AutogradError(_NONSCALAR_BACKWARD_MSG)

    root = tensor.grad_fn
    if root is None:
        # requires_grad leaf with one element: accumulate seed into itself.
        seed = _ones_like_seed(tensor)
        AccumulateGrad(tensor).apply(seed)
        return

    # Count how many times each reachable node appears as a next_function so
    # we know when its incoming gradient is complete. Also check that no
    # node's saved tensors were already released (before any apply / accumulate).
    expected: dict[int, int] = defaultdict(int)
    expected[id(root)] = 1
    seen: set[int] = {id(root)}
    reachable: list[Node] = [root]
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        node.ensure_saved_available()
        for nxt in node.next_functions:
            if nxt is None:
                continue
            nid = id(nxt)
            expected[nid] += 1
            if nid not in seen:
                seen.add(nid)
                stack.append(nxt)
                reachable.append(nxt)

    # Accumulated incoming gradient per node; pending arrivals remaining.
    from . import _engine

    pending = dict(expected)
    acc_grad: dict[int, Tensor] = {id(root): _ones_like_seed(tensor)}
    ready: deque[Node] = deque([root])

    while ready:
        node = ready.popleft()
        nid = id(node)
        grad_out = acc_grad.pop(nid)
        for next_node, grad_in in node.apply(grad_out):
            mid = id(next_node)
            if mid not in acc_grad:
                acc_grad[mid] = grad_in
            else:
                # Sum into the accumulated buffer via out-of-place engine add,
                # then keep the new tensor as the accumulator.
                acc_grad[mid] = _wrap(
                    _call_engine(
                        _engine.add, acc_grad[mid]._core, grad_in._core
                    )
                )
            pending[mid] -= 1
            if pending[mid] == 0:
                ready.append(next_node)

    if not retain_graph:
        for node in reachable:
            node.release_saved_tensors()
