
# const TRAVIS_SKIP = VERSION.minor != 4 && !isnothing(get(ENV, "TRAVIS_BRANCH", nothing))

function test_fixed_size(A, At, B, Bt, Aa, Aat, Ba, Bat)
  @test Aa * Ba ≈ A * B
  @test Aa * Bat ≈ A * Bt
  @test Aat * Ba ≈ At * B
  @test Aat * Bat ≈ At * Bt
  nothing
end

function test_fixed_size(M, K, N)
  A = @StrideArray rand(M, K)
  B = @StrideArray rand(K, N)
  At = (@StrideArray rand(K, M))'
  Bt = (@StrideArray rand(N, K))'
  Aa = Array(A)
  Ba = Array(B)
  Aat = Array(At)
  Bat = Array(Bt)
  test_fixed_size(A, At, B, Bt, Aa, Aat, Ba, Bat)
  test_fixed_size(
    StrideArrays.make_dynamic(A),
    StrideArrays.make_dynamic(At),
    B,
    Bt,
    Aa,
    Aat,
    Ba,
    Bat
  )
  test_fixed_size(
    A,
    At,
    StrideArrays.make_dynamic(B),
    StrideArrays.make_dynamic(Bt),
    Aa,
    Aat,
    Ba,
    Bat
  )
  test_fixed_size(
    StrideArrays.make_dynamic(A),
    StrideArrays.make_dynamic(At),
    StrideArrays.make_dynamic(B),
    StrideArrays.make_dynamic(Bt),
    Aa,
    Aat,
    Ba,
    Bat
  )
  nothing
end

test_fixed_size(M) = test_fixed_size(M, M, M)

# gmul(A, B) = LinearAlgebra.generic_matmatmul('N','N', A, B)

@testset "MatrixMultiply" begin
  r = 2:12
  for M ∈ r, K ∈ r, N ∈ r
    test_fixed_size(M, K, N)
  end
  r = 13:33
  for M ∈ r
    test_fixed_size(M)
  end
  M = K = N = 80
  A = @StrideArray rand(M, K)
  B = @StrideArray rand(K, N)
  C = StrideArray{Float64}(undef, (StaticInt(M), StaticInt(N)))
  M, K, N = 23, 37, 19
  @views begin
    Av = A[1:M, 1:K]
    Bv = B[1:K, 1:N]
    Cv = C[1:M, 1:N]
    Avsl = A[StaticInt(1):M, StaticInt(1):K]
    Bvsl = B[StaticInt(1):K, StaticInt(1):N]
    Cvsl = C[StaticInt(1):M, StaticInt(1):N]
    Avsr = A[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(K)]
    Bvsr = B[StaticInt(1):StaticInt(K), StaticInt(1):StaticInt(N)]
    Cvsr = C[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(N)]
  end
  Creference = Array(Av) * Array(Bvsl)
  time = @elapsed mul!(Cv, Av, Bv)
  @test Creference ≈ Cv
  @show M, K, N, time
  time = @elapsed mul!(Cvsl, Avsl, Bvsl)
  @test Creference ≈ Cv
  @show M, K, N, time
  time = @elapsed mul!(Cvsr, Avsr, Bvsr)
  @test Creference ≈ Cv
  @show M, K, N, time

  Av = A[1:M, 1:K]
  Bv = B[1:K, 1:N]
  Cv = C[1:M, 1:N]
  Avsl = A[StaticInt(1):M, StaticInt(1):K]
  Bvsl = B[StaticInt(1):K, StaticInt(1):N]
  Cvsl = C[StaticInt(1):M, StaticInt(1):N]
  Avsr = A[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(K)]
  Bvsr = B[StaticInt(1):StaticInt(K), StaticInt(1):StaticInt(N)]
  Cvsr = C[StaticInt(1):StaticInt(M), StaticInt(1):StaticInt(N)]
  Creference = Array(Av) * Array(Bvsl)
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
