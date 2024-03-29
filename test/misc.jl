
@noinline function dostuff(A; B = I)
  StrideArrays.@turbo for i in eachindex(A)
    A[i] += 1
  end
  C = A * B
  s = zero(eltype(C))
  StrideArrays.@turbo for i in eachindex(C)
    s += C[i]
  end
  s
end
function gc_preserve_test()
  A = @StrideArray rand(8, $(2 << 2))
  B = @StrideArray rand($(1 << 3), 8)
  @gc_preserve dostuff(A, B = B)
end
alloctest() = @allocated gc_preserve_test()
foo(x, f) = f(x)

@testset "Miscellaneous" begin
  x = @StrideArray rand(127)
  @test maximum(abs, x) == maximum(abs, Array(x))
  y = @StrideArray rand(3)
  @test maximum(abs, y) == maximum(abs, Array(y))
  @test iszero(alloctest())

  A = @StrideArray rand($(5 << 1), $(1 << 3))
  @test StrideArrays.static_size(A) === (StaticInt(10), StaticInt(8))
  A_u = view(A, StaticInt(1):StaticInt(6), :)
  A_l = view(A, StaticInt(7):StaticInt(10), :)
  @test A == @inferred(vcat(A_u, A_l))
  @test StrideArrays.static_size(A) === StrideArrays.static_size(vcat(A_u, A_l))

  # On 1.5 tests fail if you don't do this first.
  @test pointer(view(A, 1:StaticInt(6), :)) == pointer(A)
  A_u = view(A, 1:StaticInt(6), :)
  A_l = view(A, StaticInt(7):StaticInt(10), :)
  @test A == @inferred(vcat(A_u, A_l))
  @test StrideArrays.static_size(A) == StrideArrays.static_size(vcat(A_u, A_l))
  @test StrideArrays.static_size(A) !== StrideArrays.static_size(vcat(A_u, A_l))

  Aa = Array(A)
  @test sum(A) ≈ sum(Aa)
  @test maximum(A) == maximum(Aa) == maximum(abs, @. -A)
  @test mapreduce(abs2, +, A) ≈ mapreduce(abs2, +, Aa)

  @test A[1, 1] == A[1] == vec(A)[1]
  A[1, 1] = 3
  @test A[1, 1] == A[1] == vec(A)[1] == 3
  A[2] = 4
  @test A[2, 1] == A[2] == vec(A)[2] == 4
  vec(A)[3] = 5
  @test A[3, 1] == A[3] == vec(A)[3] == 5

  @test ArrayInterface.stride_rank(StrideArrays.similar_layout(A')) ===
        ArrayInterface.stride_rank(StrideArrays.similar_layout(Aa')) ===
        (StaticInt(2), StaticInt(1))
  @test ArrayInterface.contiguous_axis(StrideArrays.similar_layout(A')) ===
        ArrayInterface.contiguous_axis(StrideArrays.similar_layout(Aa')) ===
        StrideArrays.StaticInt{2}()

  let B = @StrideArray rand(10, 10)
    @test exp(B) ≈ exp(Array(B))
  end

  let C = @StrideArray rand(10, 10)
    Ca = Array(C)
    @test sum(C) ≈ sum(Ca)
    @test sum(view(C, :, :)) ≈ sum(C[:, :]) ≈ sum(view(Ca, :, :))
    @test sum(view(C, static(1):static(2), :)) ≈
          sum(C[static(1):static(2), :]) ≈
          sum(view(Ca, static(1):static(2), :)) ≈
          sum(view(Ca, 1:2, :))
    @test C[static(1):static(2), :] === view(C, static(1):static(2), :)
    @test view(C, :, :) === C[:, :]
  end
  let
    src1 = rand(10) .* 2 .- 1
    dst = zero(src1)
    GC.@preserve src1 dst begin
      src1_ptr = PtrArray(src1)
      dst_ptr = PtrArray(dst)
      @. dst = foo(src1, abs)
      @. dst_ptr = foo(src1_ptr, abs)
      @test dst == dst_ptr
    end
  end

  let y = Vector{Float64}(undef, 100)
    GC.@preserve y begin
      A = PtrArray(y)
      B = rand(100)
      map!(x -> x + 1, A, B)
      @test y == B .+ 1
    end
  end
end
