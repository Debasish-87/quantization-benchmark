# Profiling

The primary benchmark is intentionally not run under Nsight, because profiling
changes runtime characteristics.

After the normal benchmark identifies a bottleneck, profile selected cases only.

Recommended follow-up:
1. Pick one representative model/quantization.
2. Pick concurrency 1 for latency or 16/32 for throughput.
3. Use Nsight Systems for CPU/GPU timeline and launch/synchronization analysis.
4. Use Nsight Compute only for a small number of kernels.
5. Compare optimized vs baseline runs using the same prompt lengths and concurrency.

Do not mix profiled timings into the primary benchmark CSV.
