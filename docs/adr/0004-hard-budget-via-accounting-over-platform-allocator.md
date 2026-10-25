# The memory budget is a hard guarantee, enforced by an accounting layer over the platform allocator

Exceeding the memory budget during a real step is a bug, not an estimate error: every allocation passes through a thin accounting layer that counts it and refuses any that would cross the budget. Underneath, we use the platform allocator rather than writing our own: on Apple Silicon all memory (CPU and GPU alike, since it is unified) comes from Metal as shared-storage buffers; on Linux CPU it comes from the system allocator.

## Considered Options

- **Metal allocator with no accounting of our own**: rely on Metal's reported usage; rejected because it makes the budget advisory.
- **One Metal heap sized to the budget**: enforces the limit by construction, but makes fragmentation our problem.

## Consequences

- No tensor memory may bypass the accounting layer, including memory allocated by Python-side code paths.
- Tests assert that measured peak memory matches the memory plan's prediction within a tolerance.
