# Cascadelake

Cascadelake CPUs feature 2 512-bit-fma units, and thus can achieve high FLOPS in BLAS-like operations.
The particular CPU on which these benchmarks were run had its heavy-AVX512 clock speed set to 4.1 GHz, providing a theoretical peak of 131.2 GFLOPS/core.

Statically sized benchmarks vs [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl):
![sizedbenchmarks](../assets/sizedarraybenchmarks_cascadelake_AVX512.svg)

The `SMatrix` and `MMatrix` are the immutable and immutable matrix types from `StaticArrays.jl`, respectively, while `StrideArray.jl` and `PtrArray.jl` are mutable array types with optional static sizing providing by `StrideArrays.jl`. The benchmarks also included `jmul!` on base `Matrix{Float64}`, demonstrating the performance of StrideArrays's fully dynamic multiplication function.

`SMatrix` were only benchmarked up to size `20`x`20`. As their performance at larger sizes recently increased, I'll increase the size range at which I benchmark them in the future.


Multithreading benchmarks were run using [BLASBenchmarks.jl](https://github.com/chriselrod/BLASBenchmarks.jl):
![multithreadedbenchmarks](../assets/gemm_Float64_10_10000_cascadelake_AVX512__multithreaded_logscale.svg)

The single-threaded dynamic multiplication is competitive with `MKL` and `OpenBLAS` from around `2`x`2` to `256`x`256`:
![dgemmbenchmarkssmall](../assets/gemmFloat64_2_256_cascadelake_AVX512.svg)
However, beyond this size, performance begins to fall behind:
![dgemmbenchmarksmedium](../assets/gemmFloat64_256_2000_cascadelake_AVX512.svg)
`OpenBLAS` eventually ascends to about 120 GFLOPS, but StrideArrays seems stuck at around 100 GFLOPS.




