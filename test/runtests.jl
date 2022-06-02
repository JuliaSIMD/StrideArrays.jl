using StrideArrays, LinearAlgebra, Aqua, ArrayInterface
using Test

import InteractiveUtils
InteractiveUtils.versioninfo(stdout; verbose = true)

@show StrideArrays.VectorizationBase.register_count()
const START_TIME = time()

@time @testset "StrideArrays.jl" begin
  @test isempty(Test.detect_unbound_args(StrideArrays))
   
  @time Aqua.test_all(StrideArrays, ambiguities = false, project_toml_formatting = false, deps_compat = (VERSION <= v"1.8" || VERSION.prerelease[1] != "DEV"))
  # Currently, there is one method ambiguity:
  # - map(f::F, A::AbstractStrideArray, args::Vararg{Any, K}) where {F, K} in StrideArrays at StrideArrays/src/miscellaneous.jl:22
  # - map(f, a1::AbstractArray, a2::StaticArrays.StaticArray, as::AbstractArray...) in StaticArrays at StaticArrays/0bweZ/src/mapreduce.jl:33
  @time @test length(Test.detect_ambiguities(StrideArrays)) <= 1
  # @test isempty(detect_unbound_args(StrideArrays))
  @time include("matmul_tests.jl")
  @time include("misc.jl")
  @time include("broadcast_tests.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME) / 60
@test ELAPSED_MINUTES < 180
