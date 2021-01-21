using StrideArrays, LinearAlgebra, Aqua
using Test

import InteractiveUtils
InteractiveUtils.versioninfo(stdout; verbose = true)

@show StrideArrays.VectorizationBase.REGISTER_COUNT
const START_TIME = time()

@time @testset "StrideArrays.jl" begin
    Aqua.test_all(StrideArrays)
    # @test isempty(detect_unbound_args(StrideArrays))
    @time include("matmul_tests.jl")
    @time include("misc.jl")
    @time include("broadcast_tests.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME)/60
@test ELAPSED_MINUTES < 180
