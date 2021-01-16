# Haswell

The Haswell CPU benchmarked here is a 1.7 GHz laptop CPU. It features two 256-bit FMA units, which gives it comparable peak FLOPS/cycle to Tigerlake. But, with its smaller caches, fewer and smaller registers necessitating churning over the cache more quickly, and more limited out of order capabilities, it is much more difficult to achieve peak performance on Haswell.

Statically sized benchmarks vs [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl):
![sizedbenchmarks](../assets/sizedarraybenchmarks_haswell_AVX2.svg)

The `SMatrix` and `MMatrix` are the immutable and immutable matrix types from `StaticArrays.jl`, respectively, while `StrideArray.jl` and `PtrArray.jl` are mutable array types with optional static sizing providing by `StrideArrays.jl`. The benchmarks also included `jmul!` on base `Matrix{Float64}`, demonstrating the performance of StrideArrays's fully dynamic multiplication function.

`SMatrix` were only benchmarked up to size `20`x`20`. As their performance at larger sizes recently increased, I'll increase the size range at which I benchmark them in the future.



The fully dynamic multiplication is competitive with `MKL` and `OpenBLAS` from around `2`x`2` to `256`x`256`:
![dgemmbenchmarkssmall](../assets/gemmFloat64_2_256_haswell_AVX2.svg)
![dgemmbenchmarksmedium](../assets/gemmFloat64_256_1000_haswell_AVX2.svg)

Benchmarks will be added later.


