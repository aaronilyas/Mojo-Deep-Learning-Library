"""PyTorch-shaped ``nn.Module`` and ``nn.Parameter`` for mojodl.

Registration, named traversal, and ``__call__`` → ``forward`` match the
PyTorch CPU oracle used by ``tests/test_nn_module.py``. Linear / ReLU /
Sequential, buffers, hooks, and train/eval are intentionally out of scope.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator

from . import Tensor


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


__all__ = ["Parameter", "Module"]
