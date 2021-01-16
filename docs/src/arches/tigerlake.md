# Tigerlake

Tigerlake CPUs feature just a single 512-bit-fma unit, and thus their theoretical peak FLOPS are comparable with AVX2 CPUs featuing two 256-bit FMA units, such as Intel's Skylake or AMD's Zen2.
The much larger register file that AVX512 provides combined with its comparatively much larger L1 and L2 caches (and no doubt helped by the large out of order buffer) make it comparatively very easy to attain near peak performance on Tigerlake.

Statically sized benchmarks vs [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl):
![sizedbenchmarks](../assets/sizedarraybenchmarks_tigerlake_AVX512.svg)

The `SMatrix` and `MMatrix` are the immutable and immutable matrix types from `StaticArrays.jl`, respectively, while `StrideArray.jl` and `PtrArray.jl` are mutable array types with optional static sizing providing by `StrideArrays.jl`. The benchmarks also included `jmul!` on base `Matrix{Float64}`, demonstrating the performance of StrideArrays's fully dynamic multiplication function.

The version of `OpenBLAS` used (0.3.10) didn't support Tigerlake yet. Unlike Cascadelake, where approaching the CPU's peak performance can be challenging, it is easy with Tigerlake: Tigerlake has much larger caches and reorder buffers, making it much more capable of feeding the execution units, but has half as many of them to feed as cascadelake for these workloads (1 FMA unit vs 2 FMA units).

Threaded results of the dynamic matmul:
![threadedbenchmarks](../assets/gemm_Float64_10_10000_tigerlake_AVX512__multithreaded_logscale.svg)

Single threaded, the fully dynamic multiplication is competitive with `MKL` and `OpenBLAS` from around `2`x`2` to `256`x`256`:
![dgemmbenchmarkssmall](../assets/gemmFloat64_2_256_tigerlake_AVX512.svg)
Unlike the Cascadelake CPU, it was able to hold on with `MKL` at least through `2000`x`2000`:
![dgemmbenchmarksmedium](../assets/gemmFloat64_256_2000_tigerlake_AVX512.svg)

