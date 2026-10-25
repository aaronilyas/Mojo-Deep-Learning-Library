# Apple Silicon is the tier 1 platform; Metal is the primary compute path

The library is designed and benchmarked against an 8 GB unified-memory Apple Silicon Mac (an M2). Apple Silicon with Metal is tier 1: tuned and benchmarked against MLX. Linux on CPU is tier 2: correct and tested (it hosts the CPU reference-kernel tests), benchmarked against PyTorch. NVIDIA GPUs are deferred, and Windows is supported only through WSL2 because Mojo has no native Windows build.

## Consequences

- Peak memory is defined over a single unified pool. Adding NVIDIA later means extending the memory budget to account for separate device and host memory.
- There is no CUDA path; anyone expecting one should read this first.
