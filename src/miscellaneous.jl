

@inline Base.map!(f::F, A::AbstractStrideArray, arg1::AbstractArray, args::Vararg{AbstractArray,K}) where {F,K} = vmap!(f, A, arg1, args...)
# these two definitions are to avoid ambiguities
@inline Base.map!(f::F, A::AbstractStrideArray, arg::AbstractArray) where {F} = vmap!(f, A, arg)
@inline Base.map!(f::F, A::AbstractStrideArray, arg1::AbstractArray, arg2::AbstractArray) where {F} = vmap!(f, A, arg1, arg2)
@inline Base.map(f::F, A::AbstractStrideArray, args::Vararg{Any,K}) where {F,K} = vmap(f, A, args...)
@inline Base.reduce(op::O, A::AbstractStrideArray) where {O} = vreduce(op, A)
@inline Base.mapreduce(f::F, op::O, A::AbstractStrideArray, args::Vararg{AbstractArray,K}) where {F, O, K} = vmapreduce(f, op, A, args...)
@inline function Base.mapreduce(f::F, op::O, A::AbstractStrideArray) where {F, O}
    @gc_preserve vmapreduce(f, op, A)
end

for (op,r) ∈ ((:+,:sum), (:max,:maximum), (:min,:minimum))
    @eval begin
        @inline Base.reduce(::typeof($op), A::AbstractStrideArray; dims = nothing) = @gc_preserve vreduce($op, A, dims = dims)
        @inline Base.$r(A::AbstractStrideArray; dims = nothing) = @gc_preserve vreduce($op, A, dims = dims)
    end
end

function Base.copyto!(B::AbstractStrideArray{<:Any,<:Any,<:Any,N}, A::AbstractStrideArray{<:Any,<:Any,<:Any,N}) where {N}
    @avx for I ∈ eachindex(A, B)
        B[I] = A[I]
    end
    B
end


function maximum(::typeof(abs), A::AbstractStrideArray{S,T}) where {S,T}
    s = typemin(T)
    @avx for i ∈ eachindex(A)
        s = max(s, abs(A[i]))
    end
    s
end

function Base.vcat(A::AbstractStrideMatrix, B::AbstractStrideMatrix)
    MA, NA = size(A)
    MB, NB = size(B)
    @assert NA == NB
    TC = promote_type(eltype(A), eltype(B))
    C = StrideArray{TC}(undef, (MA + MB, NA))
    # TODO: Actually handle offsets
    @assert offsets(A) === offsets(B)
    @assert offsets(A) === offsets(C)
    @avx for j ∈ axes(A,2), i ∈ axes(A,1)
        C[i,j] = A[i,j]
    end
    @avx for j ∈ axes(B,2), i ∈ axes(B,1)
        C[i + MA,j] = B[i,j]
    end
    C
end


@inline function make_stride_dynamic(p::StridedPointer{T,N,C,B,R}) where {T,N,C,B,R}
    StridedPointer{T,N,C,B,R}(p.p, map(Int, p.strd), p.offsets)
end
@inline function make_dynamic(A::PtrArray)
    PtrArray(make_stride_dynamic(stridedpointer(A)), Base.size(A), ArrayInterface.dense_dims(A))
end
@inline function make_dynamic(A::StrideArray)
    StrideArray(make_dynamic(PtrArray(A)), A.data)
end
