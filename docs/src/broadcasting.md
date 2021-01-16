## Broadcasting

Broadcasting `StrideArrays` is also fast, e.g. to continue on the random number generation example from earlier, we could quickly calculate a Monte Carlo sample of means of log normally distributed random variables:
```julia
julia> @benchmark sum(exp.(@StrideArray randn(8,10))) # StrideArrays
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     127.652 ns (0.00% GC)
  median time:      129.033 ns (0.00% GC)
  mean time:        129.041 ns (0.00% GC)
  maximum time:     163.491 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     888

julia> @benchmark sum(exp.(@SMatrix randn(8,10))) # StaticArrays
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     678.948 ns (0.00% GC)
  median time:      690.000 ns (0.00% GC)
  mean time:        690.399 ns (0.00% GC)
  maximum time:     847.484 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     153
```


