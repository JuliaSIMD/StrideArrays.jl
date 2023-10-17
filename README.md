# StrideArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSIMD.github.io/StrideArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSIMD.github.io/StrideArrays.jl/dev)
[![Build Status](https://github.com/JuliaSIMD/StrideArrays.jl/workflows/CI/badge.svg)](https://github.com/JuliaSIMD/StrideArrays.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaSIMD/StrideArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaSIMD/StrideArrays.jl)

### Use

```julia
julia> @time using StrideArrays
  5.921865 seconds (12.17 M allocations: 722.046 MiB, 2.96% gc time, 70.89% compilation time)

julia> A = @StrideArray rand(3,4)
3×4 StrideArraysCore.StaticStrideArray{Tuple{StaticInt{3}, StaticInt{4}}, (true, true), Float64, 2, 1, 0, (1, 2), Tuple{StaticInt{8}, StaticInt{24}}, Tuple{StaticInt{1}, StaticInt{1}}, 12} with indices 1:1:3×1:1:4:
 0.504925  0.280823  0.578082  0.839807
 0.865055  0.762067  0.897201  0.593801
 0.485478  0.95566   0.439315  0.771538

julia> B = similar(A);

julia> @benchmark @. $B = log($A)
BenchmarkTools.Trial: 10000 samples with 580 evaluations.
 Range (min … max):  197.441 ns … 306.610 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     199.200 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   200.114 ns ±   2.698 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

       ▃▆██▇▄▁
  ▁▁▂▄████████▇▄▃▃▂▂▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  197 ns           Histogram: frequency by time          209 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> @benchmark sum(log.($A))
BenchmarkTools.Trial: 10000 samples with 328 evaluations.
 Range (min … max):  271.122 ns … 456.610 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     272.936 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   279.168 ns ±  17.957 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ██▆▄▂▃▃▃▂▁                                       ▁ ▁▁▁▁▁▁▁▁▁  ▂
  ███████████▆▄▁▃▁▁▁▁▃▁▁▁▃▁▁▁▄▃▅▄▄▃▄▅▆▅▆▇▆▇▆▇▇█▇███████████████ █
  271 ns        Histogram: log(frequency) by time        343 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
 ```

