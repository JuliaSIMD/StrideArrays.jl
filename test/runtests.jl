using StrideArrays, LinearAlgebra, Aqua
using Test

import InteractiveUtils
InteractiveUtils.versioninfo(stdout; verbose = true)

@show Threads.nthreads(), StrideArrays._nthreads() StrideArrays.VectorizationBase.REGISTER_COUNT
const START_TIME = time()

@inferred StrideArrays.matmul_params(Float32)
@inferred StrideArrays.matmul_params(Float64)
@inferred StrideArrays.matmul_params(Int16)
@inferred StrideArrays.matmul_params(Int32)
@inferred StrideArrays.matmul_params(Int64)

@time @testset "StrideArrays.jl" begin
    Aqua.test_all(StrideArrays)
    # @test isempty(detect_unbound_args(StrideArrays))
    @time include("matmul_tests.jl")
    @time include("misc.jl")
    @time include("broadcast_tests.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME)/60
@test ELAPSED_MINUTES < 180
