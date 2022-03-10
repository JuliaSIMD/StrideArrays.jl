## Getting Started

To install `StrideArrays.jl`, simply:
```julia
using Pkg
Pkg.add("StrideArrays")
```

This library is built on [ArrayInterface.jl](https://github.com/SciML/ArrayInterface.jl) and [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl).
It is still somewhat experimental, and many features such as good linear algebra support, are still missing. It aims to achieve high performance and provide flexibility, while keeping implementations simple.

Please file issues if you encounter problems or have feature requests.

To create an uninitialized `StrideArray`, use the constructor `StrideArray{T}(undef, size_tuple)`, e.g.:
```julia
julia> StrideArray{Float64}(undef, (3,4)) |> StrideArrays.size
(3, 4)

julia> StrideArray{Float64}(undef, (StaticInt(3),4)) |> StrideArrays.size
(Static(3), 4)

julia> StrideArray{Float64}(undef, (3,StaticInt(4))) |> StrideArrays.size
(3, Static(4))

julia> StrideArray{Float64}(undef, (StaticInt(3),StaticInt(4))) |> StrideArrays.size
(Static(3), Static(4))
```
If a size is specified by a `StaticInt`, then that dimension will be statically sized. Otherwise, it will by dynamically sized.


To create one filled with random elements, see the RNG section.



