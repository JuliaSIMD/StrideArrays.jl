using StrideArrays, LinearAlgebra, Aqua, ArrayInterface
using Test

import InteractiveUtils
InteractiveUtils.versioninfo(stdout; verbose = true)

@show StrideArrays.VectorizationBase.register_count()
const START_TIME = time()

@time @testset "StrideArrays.jl" begin
  @time Aqua.test_all(StrideArrays, ambiguities = VERSION ≥ v"1.6")
  # @test isempty(detect_unbound_args(StrideArrays))
  @time include("matmul_tests.jl")
  @time include("misc.jl")
  @time include("broadcast_tests.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME)/60
@test ELAPSED_MINUTES < 180
