
@noinline function dostuff(A; B = I)
    StrideArrays.@avx for i in eachindex(A)
        A[i] += 1
    end
    C = A * B
    s = zero(eltype(C))
    StrideArrays.@avx for i in eachindex(C)
        s += C[i]
    end
    s
end
function gc_preserve_test()
    A = @StrideArray rand(8,8);
    B = @StrideArray rand(8,8);
    @gc_preserve dostuff(A, B = B)
end

@testset "Miscellaneous" begin
    x = @StrideArray rand(127);
    @test maximum(abs, x) == maximum(abs, Array(x))
    y = @StrideArray rand(3);
    @test maximum(abs, y) == maximum(abs, Array(y))
    @test iszero(@allocated gc_preserve_test())

    A = @StrideArray rand(10,8);
    A_u = view(A, StaticInt(1):StaticInt(6), :)
    A_l = view(A, StaticInt(7):StaticInt(10), :)
    @test A == @inferred(vcat(A_u, A_l))
    @test StrideArrays.size(A) === StrideArrays.size(vcat(A_u, A_l))

    # On 1.5 tests fail if you don't do this first.
    @test pointer(view(A, 1:StaticInt(6), :)) == pointer(A)
    A_u = view(A, 1:StaticInt(6), :);
    A_l = view(A, StaticInt(7):StaticInt(10), :);
    @test A == @inferred(vcat(A_u, A_l))
    @test StrideArrays.size(A) == StrideArrays.size(vcat(A_u, A_l))
    @test StrideArrays.size(A) !== StrideArrays.size(vcat(A_u, A_l))

    Aa = Array(A)
    @test sum(A) ≈ sum(Aa)
    @test maximum(A) == maximum(Aa) == maximum(abs, @. -A)
    @test mapreduce(abs2, +, A) ≈ mapreduce(abs2, +, Aa)

    @test A[1,1] == A[1] == vec(A)[1]
    A[1,1] = 3
    @test A[1,1] == A[1] == vec(A)[1] == 3
    A[2] = 4
    @test A[2,1] == A[2] == vec(A)[2] == 4
    vec(A)[3] = 5
    @test A[3,1] == A[3] == vec(A)[3] == 5

    @test ArrayInterface.stride_rank(StrideArrays.similar_layout(A')) === ArrayInterface.stride_rank(StrideArrays.similar_layout(Aa')) === (StaticInt(2),StaticInt(1))
    @test ArrayInterface.contiguous_axis(StrideArrays.similar_layout(A')) === ArrayInterface.contiguous_axis(StrideArrays.similar_layout(Aa')) === StrideArrays.StaticInt{2}()
    
    
end

