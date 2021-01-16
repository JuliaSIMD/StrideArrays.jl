
# const TRAVIS_SKIP = VERSION.minor != 4 && !isnothing(get(ENV, "TRAVIS_BRANCH", nothing))

function test_fixed_size(M, K, N)
    A = @StrideArray rand(M,K);
    B = @StrideArray rand(K,N);
    At = (@StrideArray rand(K,M))';
    Bt = (@StrideArray rand(N,K))';
    Aa = Array(A); Ba = Array(B);
    Aat = Array(At); Bat = Array(Bt);
    time = @elapsed(@test Aa * Ba ≈ A * B)
    @show M, K, N, time
    time = @elapsed(@test Aa * Bat ≈ A * Bt)
    @show M, K, N, time
    time = @elapsed(@test Aat * Ba ≈ At * B)
    @show M, K, N, time
    time = @elapsed(@test Aat * Bat ≈ At * Bt)
    @show M, K, N, time
    nothing
end

function test_fixed_size(M)
    K = N = M
    A = @StrideArray rand(M,K);
    B = @StrideArray rand(K,N);
    Aa = Array(A); Ba = Array(B);
    time = @elapsed(@test Aa * Ba ≈ A * B)
    @show M, K, N, time
    time = @elapsed(@test Aa * Ba' ≈ A * B')
    @show M, K, N, time
    time = @elapsed(@test Aa' * Ba ≈ A' * B)
    @show M, K, N, time
    time = @elapsed(@test Aa' * Ba' ≈ A' * B')
    @show M, K, N, time
    nothing
end

# gmul(A, B) = LinearAlgebra.generic_matmatmul('N','N', A, B)

@testset "MatrixMultiply" begin
    @testset "matmul" begin
        @time for T in (Float32, Float64, Int32, Int64)
            @show T, @__LINE__
            logMmax, logKmax, logNmax = log.(StrideArrays.matmul_params(T))
            @time for logN ∈ range(0, min(logNmax + 0.7, 9.210340371976184), length = 7)
                N = round(Int, exp(logN))
                for logM ∈ range(0, logMmax + 0.7, length = 7)
                    M = round(Int, exp(logM))
                    C1 = Matrix{T}(undef, M, N);
                    C2 = similar(C1); C3 = similar(C1); C4 = similar(C1); C5 = similar(C1);
                    for logK ∈ range(0, logKmax + 0.7, length = 7)
                        K = round(Int, exp(logK))
                        
                        gopc = 2e-9M*K*N
                        A = rand(T, M, K); At = copy(A');
                        B = rand(T, K, N); Bt = copy(B');

                        blastime = @elapsed mul!(C1, A, B)
                        # blastime = @elapsed (C1 = gmul(A, B))
                        bops = gopc / blastime
                        @show (M,K,N), blastime, bops, T
                        s_time_nn = @elapsed matmul_serial!(C2, A, B); s_ops_nn = gopc / s_time_nn
                        print(stdout, "(M,K,N) = ($M,$K,$N), s_time_nn = $s_time_nn, s_ops_nn = $s_ops_nn")
                        @test C1 ≈ C2
                        s_time_nt = @elapsed matmul_serial!(C3, A, Bt'); s_ops_nt = gopc / s_time_nt
                        println(stdout, ", s_time_nt = $s_time_nt, s_ops_nt = $s_ops_nt")
                        @test C1 ≈ C3
                        if T <: AbstractFloat
                            fill!(C2, NaN); fill!(C3, NaN)
                        else
                            fill!(C2, -99999); fill!(C3, -99999)
                        end
                        t_time_nn = @elapsed matmul!(C2, A, B); t_ops_nn = gopc / t_time_nn
                        print(stdout, "(M,K,N) = ($M,$K,$N), t_time_nn = $t_time_nn, t_ops_nn = $t_ops_nn")
                        @test C1 ≈ C2
                        t_time_nt = @elapsed matmul!(C3, A, Bt'); t_ops_nt = gopc / t_time_nt
                        println(stdout, ", t_time_nt = $t_time_nt, t_ops_nt = $t_ops_nt")
                        @test C1 ≈ C3
                        
                        s_time_tn = @elapsed matmul_serial!(C4, At', B); s_ops_tn = gopc / s_time_tn
                        print(stdout, "(M,K,N) = ($M,$K,$N), s_time_tn = $s_time_tn, s_ops_tn = $s_ops_tn")
                        @test C1 ≈ C4
                        s_time_tt = @elapsed matmul_serial!(C5, At', Bt'); s_ops_tt = gopc / s_time_tt
                        println(stdout, ", s_time_tt = $s_time_tt, s_ops_tt = $s_ops_tt")
                        @test C1 ≈ C5
                        if T <: AbstractFloat
                            fill!(C4, NaN); fill!(C5, NaN)
                        else
                            fill!(C4, -99999); fill!(C5, -99999)
                        end
                        t_time_tn = @elapsed matmul!(C4, At', B); t_ops_tn = gopc / t_time_tn
                        print(stdout, "(M,K,N) = ($M,$K,$N), t_time_tn = $t_time_tn, t_ops_tn = $t_ops_tn")
                        @test C1 ≈ C4
                        t_time_tt = @elapsed matmul!(C5, At', Bt'); t_ops_tt = gopc / t_time_tt
                        println(stdout, ", t_time_tt = $t_time_tt, t_ops_tt = $t_ops_tt")
                        @test C1 ≈ C5
                    end
                end
            end
        end
    end
    @testset "FixedSize" begin
        r = 2:7
        for M ∈ r, K ∈ r, N ∈ r
            test_fixed_size(M, K, N)
        end
        r = 8:33
        for M ∈ r
            test_fixed_size(M)
        end
        M = K = N = 80
        A = @StrideArray rand(M,K);
        B = @StrideArray rand(K,N);
        C = StrideArray{Float64}(undef, (StaticInt(M),StaticInt(N)));
        M, K, N = 23, 37, 19
        @views begin
            Av = A[1:M, 1:K]; 
            Bv = B[1:K, 1:N]; 
            Cv = C[1:M, 1:N]; 
            Avsl = A[StaticInt(1):M, StaticInt(1):K]; 
            Bvsl = B[StaticInt(1):K, StaticInt(1):N]; 
            Cvsl = C[StaticInt(1):M, StaticInt(1):N]; 
            Avsr = A[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(K)]; 
            Bvsr = B[StaticInt(1):StaticInt(K), StaticInt(1):StaticInt(N)]; 
            Cvsr = C[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(N)]; 
        end
        Creference = Array(Av) * Array(Bvsl);
        time = @elapsed mul!(Cv, Av, Bv)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsl, Avsl, Bvsl)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsr, Avsr, Bvsr)
        @test Creference ≈ Cv
        @show M, K, N, time

        Av = A[1:M, 1:K]; 
        Bv = B[1:K, 1:N]; 
        Cv = C[1:M, 1:N]; 
        Avsl = A[StaticInt(1):M, StaticInt(1):K]; 
        Bvsl = B[StaticInt(1):K, StaticInt(1):N]; 
        Cvsl = C[StaticInt(1):M, StaticInt(1):N]; 
        Avsr = A[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(K)]; 
        Bvsr = B[StaticInt(1):StaticInt(K), StaticInt(1):StaticInt(N)]; 
        Cvsr = C[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(N)]; 
        Creference = Array(Av) * Array(Bvsl);
        time = @elapsed mul!(Cv, Av, Bv)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsl, Avsl, Bvsl)
        @test Creference ≈ Cv
        @show M, K, N, time
        time = @elapsed mul!(Cvsr, Avsr, Bvsr)
        @test Creference ≈ Cv
        @show M, K, N, time

    end
end


