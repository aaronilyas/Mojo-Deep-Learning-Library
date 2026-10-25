# Mojo Deep Learning Library

A memory-efficient deep learning library for local AI development on constrained hardware: training, fine-tuning, and running models on a single small machine.

## Language

**Step**:
The unit of work the library budgets: forward, backward, and optimizer update for training; one forward pass for inference.
_Avoid_: Iteration, batch

**Target device**:
An Apple Silicon Mac with 8 GB of unified memory; the machine every design decision is judged against.
_Avoid_: Test machine, dev box

**Peak memory**:
The highest total memory held by the process during one **Step**, counting CPU and GPU together since unified memory is one pool.
_Avoid_: Memory usage, RAM usage, footprint

**Memory budget**:
A user-set upper bound on **Peak memory**; the library plans each step to fit under it and refuses to run a step it cannot fit.
_Avoid_: Memory limit, memory cap, quota

**Baseline**:
The incumbent framework that **Peak memory** is compared against on the same device and step: MLX on Apple Silicon, PyTorch elsewhere.
_Avoid_: Reference, competitor

**Dry run**:
A trace of a **Step** on **Meta tensors**, used to compute its **Peak memory** and fit it to the **Memory budget** before anything is allocated.
_Avoid_: Simulation, profiling pass

**Meta tensor**:
A tensor that carries shape and dtype but no data, as in PyTorch's `"meta"` device.
_Avoid_: Fake tensor, shape-only tensor

**Reference kernel**:
The plain CPU implementation of an operation, used as ground truth when testing faster kernels.
_Avoid_: CPU fallback, naive kernel

**Platform tier**:
The level of support a platform gets: tier 1 is tuned and benchmarked (Apple Silicon with Metal); tier 2 is correct and tested but not tuned (Linux CPU).
_Avoid_: Support level, backend priority

**Memory-saving technique**:
One way the library trades compute or precision for memory: activation checkpointing, weight quantization, compact optimizer state, gradient accumulation, and (later) offloading to disk.
_Avoid_: Optimization, strategy, lever

**Memory plan**:
The result of a **Dry run**: predicted **Peak memory**, its breakdown by layer and category, and the **Memory-saving techniques** chosen per layer.
_Avoid_: Schedule, allocation plan, report

**Pin**:
A user's manual choice of a **Memory-saving technique** for part of the model, which the **Memory plan** must respect.
_Avoid_: Override, constraint, hint

**Shape bound**:
A user-declared upper limit on a dimension whose size depends on the data, which the **Dry run** plans for as the worst case.
_Avoid_: Max shape, dynamic dim hint
