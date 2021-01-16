using StrideArrays

@testset "Broadcast" begin
    M, K, N = 47, 85, 74
    # for T ∈ (Float32, Float64)
    A = @StrideArray randn(13,29);
    b = @StrideArray rand(13);
    c = @StrideArray rand(29);
    D = @. exp(A) + b * log(c');

    Aa = Array(A); ba = Array(b); ca = Array(c);
    Da = @. exp(Aa) + ba * log(ca');

    @test D ≈ Da
    # end
end

