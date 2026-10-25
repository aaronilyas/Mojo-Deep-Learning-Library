"""Eager autograd: graph nodes, backward traversal, per-op backward rules.

Recording is eager (ops push nodes as they run). Nodes reference the nodes of
their inputs, not the input tensors, and keep alive only tensors the backward
formula needs (saved tensors). Whole-tensor sum saves none — only shape and
device. Leaves are reached through AccumulateGrad, which holds the leaf.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Tensor


# device_code values match the engine / Python veneer.
_DEVICE_CPU = 0
_DEVICE_META = 1


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


class Node:
    """Base class for backward graph nodes."""

    # Subclasses set next_functions to a tuple of upstream Node | None.
    next_functions: tuple

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
    """Whole-tensor sum: expand the scalar upstream gradient to the input shape.

    Saves no tensors — only the input shape and device.
    """

    __slots__ = ("next_functions", "input_shape", "input_device")

    def __init__(
        self,
        input_node: Node,
        input_shape: tuple,
        input_device: str,
    ):
        self.next_functions = (input_node,)
        self.input_shape = input_shape
        self.input_device = input_device

    def apply(self, grad_output: Tensor) -> list[tuple[Node, Tensor]]:
        from . import _engine

        expanded = _wrap(
            _call_engine(_engine.expand_scalar, grad_output._core, self.input_shape)
        )
        return [(self.next_functions[0], expanded)]


def input_node(tensor: Tensor) -> Node | None:
    """Node to wire as the upstream of an op that used `tensor`.

    Non-grad tensors contribute nothing. Results with a grad_fn reuse that
    node. Leaves get a fresh AccumulateGrad that holds the leaf (the leaf
    itself does not point at the node, so there is no reference cycle).
    """
    if not tensor.requires_grad:
        return None
    if tensor.grad_fn is not None:
        return tensor.grad_fn
    return AccumulateGrad(tensor)


def attach(result: Tensor, grad_fn: Node | None) -> Tensor:
    """Shared recording helper: every Tensor op calls this with its result node.

    When `grad_fn` is None the result is left as a non-grad tensor. Otherwise
    the result requires grad, is not a leaf, and carries `grad_fn`.
    """
    if grad_fn is None:
        return result
    result._requires_grad = True
    result._grad_fn = grad_fn
    return result


def record_sum(result: Tensor, inp: Tensor) -> Tensor:
    """Attach a SumBackward node when `inp` requires grad."""
    nxt = input_node(inp)
    if nxt is None:
        return result
    return attach(result, SumBackward(nxt, inp.shape, inp.device))


def _ones_like_seed(tensor: Tensor) -> Tensor:
    """Engine-created ones tensor with the output's shape and device (seed grad)."""
    from . import _engine

    return _wrap(
        _call_engine(
            _engine.ones, tensor.shape, _device_code(tensor.device)
        )
    )


def backward(tensor: Tensor) -> None:
    """Run reverse-mode AD from `tensor` (assumes a 1-element output for now).

    Visits nodes in reverse topological order. Each node's incoming gradients
    are fully accumulated before its `apply` runs, so a tensor used more than
    once receives the summed gradient.
    """
    root = tensor.grad_fn
    if root is None:
        # Leaf or non-grad tensor: nothing to propagate through a graph.
        # A requires_grad leaf scalar would accumulate a seed into itself; the
        # current tests always call backward on a sum result.
        if tensor.requires_grad:
            seed = _ones_like_seed(tensor)
            AccumulateGrad(tensor).apply(seed)
        return

    # Count how many times each reachable node appears as a next_function so
    # we know when its incoming gradient is complete.
    expected: dict[int, int] = defaultdict(int)
    expected[id(root)] = 1
    seen: set[int] = {id(root)}
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        for nxt in node.next_functions:
            if nxt is None:
                continue
            nid = id(nxt)
            expected[nid] += 1
            if nid not in seen:
                seen.add(nid)
                stack.append(nxt)

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
