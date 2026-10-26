"""PyTorch-shaped ``nn.Module``, ``nn.Parameter``, and basic layers for mojodl.

Registration, named traversal, and ``__call__`` → ``forward`` match the
PyTorch CPU oracle. ``Linear``, ``ReLU``, and ``Sequential`` compose existing
autograd-tracked ops (no new engine kernels). Buffers, hooks, and train/eval
are intentionally out of scope.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import Iterator

import numpy as np

from . import Tensor, no_grad, ones, rand, zeros


class StateDictError(ValueError):
    """Raised when ``load_state_dict`` cannot apply a state mapping.

    The message names every offending parameter (unexpected, missing,
    non-float32-convertible, or shape-mismatched).
    """


class Parameter(Tensor):
    """A Tensor registered as a learnable Module weight.

    Wraps existing ``Tensor`` engine storage without copying. Array-like
    input is converted to a ``Tensor`` first. Always a leaf
    (``grad_fn is None``, ``grad is None``); ``requires_grad`` defaults
    to True.
    """

    def __init__(self, data, requires_grad: bool = True):
        if isinstance(data, Tensor):
            # Share engine storage; do not allocate or copy.
            self._core = data._core
            self._requires_grad = bool(requires_grad)
            self._grad = None
            self._grad_fn = None
        else:
            # Build a Tensor first, then take ownership of its storage.
            t = Tensor(data)
            self._core = t._core
            self._requires_grad = bool(requires_grad)
            self._grad = None
            self._grad_fn = None


class Module:
    """Base class for all neural network modules.

    Assigning a ``Parameter`` or ``Module`` to an attribute registers it.
    Other values (including plain ``Tensor``) are normal attributes.
    ``named_parameters`` / ``named_modules`` share one depth-first
    registration-order traversal with identity dedup (first name wins).
    """

    def __init__(self) -> None:
        # Use object.__setattr__ so the overridden __setattr__ can run.
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())

    def __getattr__(self, name: str):
        # Look up registered parameters and sub-modules. Guard with
        # ``__dict__`` so a Module used before ``super().__init__()``
        # raises AttributeError instead of recursing.
        if "_parameters" in self.__dict__:
            params = self.__dict__["_parameters"]
            if name in params:
                return params[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value) -> None:
        # Internal bookkeeping keys bypass registration.
        if name in ("_parameters", "_modules"):
            object.__setattr__(self, name, value)
            return

        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if params is None or modules is None:
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call"
            )

        def _remove(d: OrderedDict) -> None:
            if name in d:
                del d[name]

        if isinstance(value, Parameter):
            _remove(modules)
            if name in self.__dict__:
                del self.__dict__[name]
            # Re-assigning an existing name replaces in place (same position).
            params[name] = value
        elif isinstance(value, Module):
            if value is self:
                raise AttributeError("cannot assign self as a sub-module")
            _remove(params)
            if name in self.__dict__:
                del self.__dict__[name]
            modules[name] = value
        else:
            # Non-Parameter / non-Module clears any prior registration.
            _remove(params)
            _remove(modules)
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if params is not None and name in params:
            del params[name]
            return
        if modules is not None and name in modules:
            del modules[name]
            return
        object.__delattr__(self, name)

    def __call__(self, *args, **kwargs):
        """Single entry point for a forward pass (hook point for Dry run)."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__}.forward() is not implemented"
        )

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Immediate sub-modules in registration order."""
        modules = self.__dict__.get("_modules")
        if modules is None:
            raise AttributeError(
                "cannot call named_children before Module.__init__() call"
            )
        for name, module in modules.items():
            if module is not None:
                yield name, module

    def children(self) -> Iterator[Module]:
        for _, module in self.named_children():
            yield module

    def named_modules(
        self,
        memo: set | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Module]]:
        """Yield ``(qualified_name, module)`` depth-first in registration order.

        The first yield is ``("", self)`` when ``prefix`` is empty. Sub-module
        names are dot-qualified. Deduplicated by module identity when
        ``remove_duplicate`` is True (first name wins).
        """
        if memo is None:
            memo = set()
        if self in memo:
            return
        if remove_duplicate:
            memo.add(self)
        yield prefix, self
        for name, module in self.named_children():
            if prefix:
                child_prefix = prefix + "." + name
            else:
                child_prefix = name
            yield from module.named_modules(
                memo, child_prefix, remove_duplicate
            )

    def _named_members(
        self,
        get_members_fn,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, object]]:
        """Shared traversal: walk ``named_modules``, then each module's members."""
        memo: set = set()
        if recurse:
            modules: Iterator[tuple[str, Module]] = self.named_modules(
                prefix=prefix, remove_duplicate=remove_duplicate
            )
        else:
            modules = iter([(prefix, self)])
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                if module_prefix:
                    name = module_prefix + "." + k
                else:
                    name = k
                yield name, v

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Parameter]]:
        """Yield ``(qualified_name, Parameter)`` depth-first; each object once."""

        def _params(module: Module):
            params = module.__dict__.get("_parameters")
            if params is None:
                return ()
            return params.items()

        yield from self._named_members(
            _params,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def state_dict(self) -> OrderedDict[str, object]:
        """Export Parameters as fresh float32 NumPy copies keyed by qualified name.

        Order matches ``named_parameters()``. Each value is an independent
        ``numpy`` array; mutating it does not change the Module.
        """
        return OrderedDict(
            (name, param.numpy()) for name, param in self.named_parameters()
        )

    def load_state_dict(self, state) -> None:
        """Import Parameter values from a name→array mapping.

        Keys must match ``named_parameters()`` exactly. Validates all entries
        (unexpected / missing names, float32 convertibility, shapes) before
        modifying any Parameter; on failure raises one ``StateDictError``
        naming every offender and allocates no engine storage. On success,
        replaces each Parameter's engine storage with a fresh float32 copy
        while preserving Parameter identity, ``requires_grad``, ``.grad``,
        and leaf status (no autograd graph is recorded).
        """
        if not isinstance(state, Mapping):
            raise TypeError(
                f"load_state_dict(): state must be a Mapping, "
                f"got {type(state).__name__}"
            )

        params = OrderedDict(self.named_parameters())
        param_names = set(params)
        state_names = set(state.keys())

        problems: list[str] = []
        unexpected = sorted(state_names - param_names)
        missing = sorted(param_names - state_names)
        if unexpected:
            problems.append(
                "Unexpected key(s): " + ", ".join(unexpected)
            )
        if missing:
            problems.append("Missing key(s): " + ", ".join(missing))

        # Validate overlapping names with NumPy only — no engine allocation
        # until every check passes (atomic rejection).
        converted: dict[str, np.ndarray] = {}
        for name in sorted(param_names & state_names):
            param = params[name]
            value = state[name]
            try:
                arr = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                problems.append(f"cannot convert {name} to float32")
                continue
            if arr.dtype == object:
                problems.append(f"cannot convert {name} to float32")
                continue
            if tuple(arr.shape) != tuple(param.shape):
                problems.append(
                    f"size mismatch for {name}: "
                    f"expected {tuple(param.shape)}, got {tuple(arr.shape)}"
                )
                continue
            converted[name] = arr

        if problems:
            raise StateDictError(
                "Error(s) in loading state_dict:\n" + "\n".join(problems)
            )

        # Validation succeeded — copy into existing Parameter objects.
        for name, param in params.items():
            with no_grad():
                fresh = Tensor(converted[name])
            param._core = fresh._core


def _require_nonneg_int(name: str, value) -> int:
    """Validate a Linear size argument: non-negative int, bool rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"Linear(): {name} must be an int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"Linear(): {name} must be non-negative, got {value}")
    return value


def _uniform_parameter(shape, bound: float, device: str, seed):
    """Parameter ~ U(-bound, bound) from engine ``rand``, under ``no_grad``.

    Meta keeps the affine on Meta tensors so construction allocates nothing.
    CPU builds scalar scale/shift Tensors and maps U(0, 1) → U(-bound, bound).
    """
    with no_grad():
        u = rand(shape, device=device, seed=seed)
        if device == "meta":
            scale = ones((), device="meta")
            shift = zeros((), device="meta")
        else:
            scale = Tensor([2.0 * bound])
            shift = Tensor([-bound])
        data = u * scale + shift
    return Parameter(data)


class Linear(Module):
    """Affine transform: ``y = x @ weight.T (+ bias)`` for 2-D ``x``.

    Weight shape is ``(out_features, in_features)``; bias shape is
    ``(out_features,)`` when ``bias=True``. Init matches PyTorch's default
    uniform bound ``1/sqrt(in_features)``. Forward uses existing matmul,
    transpose, and add so autograd tracks weight, bias, and input.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        *,
        device: str = "cpu",
        seed=None,
    ):
        super().__init__()
        in_features = _require_nonneg_int("in_features", in_features)
        out_features = _require_nonneg_int("out_features", out_features)
        self.in_features = in_features
        self.out_features = out_features
        bound = (
            1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        )
        self.weight = _uniform_parameter(
            (out_features, in_features), bound, device, seed
        )
        if bias:
            # Distinct stream from weight when seed is an int; fresh draw if None.
            bias_seed = None if seed is None else (seed + 1) % (2**64)
            self.bias = _uniform_parameter(
                (out_features,), bound, device, bias_seed
            )
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """Elementwise ReLU module; no parameters."""

    def forward(self, x):
        return x.relu()


class Sequential(Module):
    """Chains child modules in registration order under names ``\"0\"``, ``\"1\"``, ..."""

    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            setattr(self, str(i), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


__all__ = [
    "Parameter",
    "Module",
    "Linear",
    "ReLU",
    "Sequential",
    "StateDictError",
]
