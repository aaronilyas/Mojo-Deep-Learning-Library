# Eager, PyTorch-style API; the memory budget is enforced by a dry run

The Python API executes eagerly and mirrors PyTorch (`Tensor`, `nn.Module`, `.backward()`), because that is the API users and coding agents already know best. To refuse a step that would exceed the memory budget before allocating anything, the step is first traced as a dry run on meta tensors; the planner computes its peak memory, chooses memory-saving techniques to fit, and only then runs it for real.

## Considered Options

- **Lazy evaluation (as in MLX)**: the planner sees the whole step for free, but the API is less familiar and errors surface far from the line that caused them.
- **Static define-then-run graph**: easiest to plan, worst ergonomics; rejected.

## Consequences

- Every operation needs a meta-tensor implementation (shape, dtype, and memory accounting) alongside its real kernels.
- Steps whose shapes depend on data values cannot be planned exactly; users declare a shape bound per data-dependent dimension, the dry run plans for the worst case, and exceeding the bound raises instead of overshooting the budget.
