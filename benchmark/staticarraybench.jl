
using StrideArrays, StaticArrays, LinearAlgebra, BenchmarkTools

# BLAS.set_num_threads(1)

# For laptops that thermally throttle, you can set the `JULIA_SLEEP_BENCH` environment variable for #seconds to sleep before each `@belapsed`
const SLEEPTIME = parse(Float64, get(ENV, "JULIA_SLEEP_BENCH", "0"))
maybe_sleep() = iszero(SLEEPTIME) || sleep(SLEEPTIME)
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 1_000_000
# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

matrix_sizes(x::Integer) = (x,x,x)
matrix_sizes(x::NTuple{3}) = x

function runbenches(sr, ::Type{T} = Float64) where {T}
    bench_results = Matrix{T}(undef, length(sr), 5)
    for (i,s) ∈ enumerate(sr)
        M, N, K = matrix_sizes(s)
        if true#s ≤ 20
            Astatic = @SMatrix rand(T, M, K);
            Bstatic = @SMatrix rand(T, K, N);
            maybe_sleep()
            bench_results[i,1] = @belapsed $(Ref(Astatic))[] * $(Ref(Bstatic))[]
            Amutable = MArray(Astatic);
            Bmutable = MArray(Bstatic);
            Cmutable = similar(Amutable);
        else
            bench_results[i,1] = Inf
            Amutable = MMatrix{M,K,T}(undef)
            Bmutable = MMatrix{K,N,T}(undef)
            Cmutable = MMatrix{M,N,T}(undef)
            @inbounds for i ∈ 1:s^2
                Amutable[i] = rand()
                Bmutable[i] = rand()
            end
        end
        maybe_sleep()
        bench_results[i,2] = @belapsed mul!($Cmutable, $Amutable, $Bmutable)
        Afixed = StrideArray(Amutable)
        Bfixed = StrideArray(Bmutable)
        Cfixed = StrideArray{T}(undef, (StaticInt(M),StaticInt(N)))
        maybe_sleep()
        bench_results[i,3] = @belapsed mul!($Cfixed, $Afixed, $Bfixed)
        Cfixed2 = similar(Cfixed);
        Aptr = PtrArray(Afixed); Bptr = PtrArray(Bfixed); Cptr = PtrArray(Cfixed2);
        maybe_sleep()
        GC.@preserve Afixed Bfixed Cfixed2 begin
            bench_results[i,4] = @belapsed mul!($Cptr, $Aptr, $Bptr)
        end
        A = Array(Afixed); B = Array(Bfixed); C = Matrix{T}(undef, M, N);
        maybe_sleep()
        bench_results[i,5] = @belapsed jmul!($C, $A, $B)
        # @show Array(Cmutable) Cfixed Cfixed2 C
        @assert Array(Cmutable) ≈ Cfixed ≈ Cfixed2 ≈ C
        v = @view(bench_results[i,:])'
        @show s, v
    end
    bench_results
end

sizerange = 2:48
br = runbenches(sizerange);
using DataFrames, VegaLite

gflops = @. 2e-9 * (sizerange) ^ 3 / br;

df = DataFrame(gflops);
matmulmethodnames = [:SMatrix, :MMatrix, :StrideArray, :PtrArray, :matmul!];
rename!(df, matmulmethodnames);
df.Size = sizerange

function pick_suffix(desc = "")
    suffix = if StrideArrays.VectorizationBase.AVX512F
        "AVX512"
    elseif StrideArrays.VectorizationBase.AVX2
        "AVX2"
    elseif StrideArrays.VectorizationBase.REGISTER_SIZE == 32
        "AVX"
    else
        "REGSUZE$(StrideArrays.VectorizationBase.REGISTER_SIZE)"
    end
    if desc != ""
        suffix *= '_' * desc
    end
    "$(Sys.CPU_NAME)_$suffix"
end

dfs = stack(df, matmulmethodnames, variable_name = :MatMulType, value_name = :GFLOPS);
p = dfs |> @vlplot(:line, x = :Size, y = :GFLOPS, width = 900, height = 600, color = {:MatMulType});
save(joinpath(pkgdir(StrideArrays), "docs/src/assets/sizedarraybenchmarks_$(pick_suffix()).svg"), p)



