using StrideArrays, Test

@testset "Broadcast" begin
  M, K, N = 47, 85, 74
  # for T ∈ (Float32, Float64)
  A = @StrideArray randn(13, 29)
  b = @StrideArray rand(13)
  c = @StrideArray rand(29)
  D = @. exp(A) + b * log(c')

  Aa = Array(A)
  ba = Array(b)
  ca = Array(c)
  Da = @. exp(Aa) + ba * log(ca')

  @test D ≈ Da
  A .= zero(eltype(A))
  @test all(==(0), A)

  

  u1 = StrideArray(ones(1, 10), (static(1), 10));
  u2 = StrideArray(collect(0:2:18)', (static(1), 10));
  u3 = StrideArray(ones(2, 10), (static(2), 10));

  @views u1[:, 1] .= u2[:, 1]
  @views u3[:, 1] .= u2[:, 1]
  @test u1[1] == 0
  @test all(isone, @view(u1[1,2:end]))
  @test u3[1,1] == 0
  @test u3[2,1] == 0
  @test all(isone, @view(u3[:,2:end]))
  
  # end
end
