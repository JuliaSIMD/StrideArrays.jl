## Random Number Generation

Randomly generating `StrideArrays` is fast, and can be done via a convenient macro:
```julia
julia> using StrideArrays, StaticArrays, BenchmarkTools

julia> @btime sum(@StrideArray randn(8,10)) # StrideArrays
  103.613 ns (0 allocations: 0 bytes)
18.015335007499978

julia> @btime sum(@SMatrix randn(8,10)) # StaticArrays
  297.042 ns (0 allocations: 0 bytes)
-4.091586809768035

julia> @btime sum(@StrideArray rand(8,10)) # StrideArrays
  18.862 ns (0 allocations: 0 bytes)
43.61560492320911

julia> @btime sum(@SMatrix rand(8,10)) # StaticArrays
  171.001 ns (0 allocations: 0 bytes)
38.47263930206726
```


