## Stack Allocation

Stack allocated arrays are great, as are mutable arrays.

`StrideArrays.jl` tries it's hardest to provide you with both. As you may have noted from the RNG and broadcasting pages, we were creating mutable `StrideArray`s without suffering memory allocations, just like with the immutable `StaticArrays.SArray` type. The mutable `StaticArrays.MArray`, on the other hand, would have allocated:
```julia
julia> @benchmark sum(exp.(@StrideArray randn(8,10))) # StrideArrays
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     127.557 ns (0.00% GC)
  median time:      127.986 ns (0.00% GC)
  mean time:        128.116 ns (0.00% GC)
  maximum time:     165.890 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     888

julia> @benchmark sum(exp.(@MMatrix randn(8,10))) # StaticArrays
BenchmarkTools.Trial:
  memory estimate:  672 bytes
  allocs estimate:  1
  --------------
  minimum time:     703.599 ns (0.00% GC)
  median time:      862.130 ns (0.00% GC)
  mean time:        887.160 ns (4.56% GC)
  maximum time:     136.675 μs (99.29% GC)
  --------------
  samples:          10000
  evals/sample:     142
```

This is achieved thanks to a convenient macro, `StrideArrays.@gc_preserve`. When the macro is applied to a function call, it `GC.@preserve`s all the arrays, and substitutes them with `PtrArray`s.
This will safely preserve the array's memory during the call, while promising that the array won't escape, so that it may be stack allocated. Otherwise, passing `mutable struct`s to non-inlined functions currently forces heap allocation.
Many functions are overloaded for `StrideArray`s to provide a `@gc_preserve` barrier, so that calling them will not force heap allocation. However, doing this systematically is still a work in progress, so please file an issue if you encounter a function commonly used on arrays, especially if already defined in `StrideArrays.jl`, in which this is not the case.

When writing code making use of statically sized `StrideArray`s, you can use `@gc_preserve` in your own code when you can promise the array won't escape to make use of mutable stack allocated arrays.
Note that `@gc_preserve` should also work on `MArray`s.



