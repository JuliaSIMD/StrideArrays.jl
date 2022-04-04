
@inline LinearAlgebra.mul!(C::AbstractStrideMatrix, A::StridedMatrix, B::StridedMatrix) =
    (@gc_preserve(matmul!(C, A, B)); return C)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::AbstractStrideMatrix, B::StridedMatrix) =
    (@gc_preserve(matmul!(C, A, B)); return C)
@inline LinearAlgebra.mul!(C::StridedMatrix, A::StridedMatrix, B::AbstractStrideMatrix) =
    (@gc_preserve(matmul!(C, A, B)); return C)
@inline LinearAlgebra.mul!(
    C::AbstractStrideMatrix,
    A::AbstractStrideMatrix,
    B::StridedMatrix,
) = (@gc_preserve(matmul!(C, A, B)); return C)
@inline LinearAlgebra.mul!(
    C::AbstractStrideMatrix,
    A::StridedMatrix,
    B::AbstractStrideMatrix,
) = (@gc_preserve(matmul!(C, A, B)); return C)
@inline LinearAlgebra.mul!(
    C::StridedMatrix,
    A::AbstractStrideMatrix,
    B::AbstractStrideMatrix,
) = (@gc_preserve(matmul!(C, A, B)); return C)
@inline LinearAlgebra.mul!(
    C::AbstractStrideMatrix,
    A::AbstractStrideMatrix,
    B::AbstractStrideMatrix,
) = (@gc_preserve(matmul!(C, A, B)); return C)

@inline function stridematmul(
    A::AbstractStrideMatrix{<:Any,<:Any,TA},
    B::AbstractStrideMatrix{<:Any,<:Any,TB},
) where {TA<:Base.HWReal,TB<:Base.HWReal}
    M, KA = size(A)
    KB, N = size(B)
    @assert KA == KB "Size mismatch."
    K = Octavian._select(KA, KB)
    TC = promote_type(TA, TB)
    C = StrideArray{TC}(undef, (M, N))
    @gc_preserve matmul!(
        C,
        A,
        B,
        One(),
        Zero(),
        nothing,
        (M, K, N),
        ArrayInterface.contiguous_axis(C),
    )
    return C
end
@inline Base.:*(
    A::AbstractStrideMatrix{<:Any,<:Any,TA},
    B::StridedMatrix{TB},
) where {TA<:Base.HWReal,TB<:Base.HWReal} = @gc_preserve(stridematmul(A, B))
@inline Base.:*(
    A::AbstractStrideMatrix{<:Any,<:Any,TA},
    B::StridedMatrix{TB},
) where {TA<:Union{Float32,Float64},TB<:Union{Float32,Float64}} =
    @gc_preserve(stridematmul(A, B))
@inline Base.:*(
    A::StridedMatrix{TA},
    B::AbstractStrideMatrix{<:Any,<:Any,TB},
) where {TA<:Base.HWReal,TB<:Base.HWReal} = @gc_preserve(stridematmul(A, B))
@inline Base.:*(
    A::StridedMatrix{TA},
    B::AbstractStrideMatrix{<:Any,<:Any,TB},
) where {TA<:Union{Float32,Float64},TB<:Union{Float32,Float64}} =
    @gc_preserve(stridematmul(A, B))
@inline Base.:*(
    A::AbstractStrideMatrix{<:Any,<:Any,TA},
    B::AbstractStrideMatrix{<:Any,<:Any,TB},
) where {TA<:Base.HWReal,TB<:Base.HWReal} = @gc_preserve(stridematmul(A, B))
@inline Base.:*(
    A::AbstractStrideMatrix{<:Any,<:Any,TA},
    B::AbstractStrideMatrix{<:Any,<:Any,TB},
) where {TA<:Union{Float32,Float64},TB<:Union{Float32,Float64}} =
    @gc_preserve(stridematmul(A, B))

@inline extract_λ(a) = a
@inline extract_λ(a::UniformScaling) = a.λ
@inline function Base.:*(
    A::AbstractStrideArray{S,D,T},
    bλ::Union{Tb,UniformScaling{Tb}},
) where {S,D,T<:VectorizationBase.NativeTypes,Tb<:Real}
    mv = similar(A)
    b = T(extract_λ(bλ))
    @avx for i ∈ eachindex(A)
        mv[i] = A[i] * b
    end
    mv
end

function LinearAlgebra.mul!(
    C::AbstractStrideMatrix{<:Any,<:Any,T},
    A::LinearAlgebra.Diagonal{T,<:AbstractStrideVector{<:Any,T}},
    B::StridedMatrix{T},
) where {T}
    M, K, N = matmul_axes(C, A, B)
    MandK = ArrayInterface._pick_range(M, K)
    vA = parent(A)
    @avx for n ∈ N, m ∈ 1:MandK
        C[m, n] = vA[m] * B[m, n]
    end
    C
end
@inline function Base.:*(
    A::LinearAlgebra.Diagonal{T,<:AbstractVector{T}},
    B::AbstractStrideMatrix{<:Any,<:Any,T},
) where {T}
    mul!(similar(B), A, B)
end
