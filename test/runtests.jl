using StrideArrays, LinearAlgebra, Aqua, ArrayInterface
using Test

import InteractiveUtils
InteractiveUtils.versioninfo(stdout; verbose = true)

@show StrideArrays.VectorizationBase.register_count()
const START_TIME = time()

@time @testset "StrideArrays.jl" begin
  @test isempty(Test.detect_unbound_args(StrideArrays))

  @time Aqua.test_all(
    StrideArrays,
    ambiguities = false,
    project_toml_formatting = false,
    deps_compat = VERSION <= v"1.8" || isempty(VERSION.prerelease),
  )
  # Currently, there are five method ambiguities:
  # (rand!(A::AbstractStrideArray, args::Vararg{Any, K}) where K in StrideArrays at StrideArrays/src/rand.jl:3, rand!(f::F, rng::VectorizedRNG.AbstractVRNG, x::AbstractArray{T}, α::Number, β, γ) where {T<:Union{Float32, Float64}, F} in VectorizedRNG at VectorizedRNG/L3orR/src/api.jl:242)
  # (map(f::F, A::AbstractStrideArray, args::Vararg{Any, K}) where {F, K} in StrideArrays at StrideArrays/src/miscellaneous.jl:37, map(f, a1::AbstractArray, a2::StaticArraysCore.StaticArray, as::AbstractArray...) in StaticArrays at StaticArrays/B0HhH/src/mapreduce.jl:33)
  # (rand!(A::AbstractStrideArray, args::Vararg{Any, K}) where K in StrideArrays at StrideArrays/src/rand.jl:3, rand!(f::F, rng::VectorizedRNG.AbstractVRNG, x::AbstractArray{T}, α::Number, β) where {T<:Union{Float32, Float64}, F} in VectorizedRNG at VectorizedRNG/L3orR/src/api.jl:242)
  # (rand!(A::AbstractStrideArray, args::Vararg{Any, K}) where K in StrideArrays at StrideArrays/src/rand.jl:3, rand!(f::F, rng::VectorizedRNG.AbstractVRNG, x::AbstractArray{T}) where {T<:Union{Float32, Float64}, F} in VectorizedRNG at VectorizedRNG/L3orR/src/api.jl:242)
  # (rand!(A::AbstractStrideArray, args::Vararg{Any, K}) where K in StrideArrays at StrideArrays/src/rand.jl:3, rand!(f::F, rng::VectorizedRNG.AbstractVRNG, x::AbstractArray{T}, α::Number) where {T<:Union{Float32, Float64}, F} in VectorizedRNG at VectorizedRNG/L3orR/src/api.jl:242)
  ambiguities = Test.detect_ambiguities(StrideArrays)
  @show ambiguities
  @time @test length(ambiguities) == 0
  @time include("matmul_tests.jl")
  @time include("misc.jl")
  @time include("broadcast_tests.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME) / 60
@test ELAPSED_MINUTES < 180
