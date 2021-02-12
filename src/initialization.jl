
# @generated default_offsets(::Val{N}) where {N} = :(Base.Cartesian.@ntuple $N n -> One())

function dense_quote(N::Int, b::Bool)
    d = Expr(:tuple)
    for n in 1:N
        push!(d.args, b)
    end
    Expr(:call, Expr(:curly, :Val, d))
end
@generated all_dense(::Val{N}) where {N} = dense_quote(N, true)
# @generated none_dense(::Val{N}) where {N} = dense_quote(N, false)

# @generated function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}, x::Tuple{Vararg{Integer,N}}) where {T,N}
#     q = Expr(:block, Expr(:meta,:inline))
    
# end
@inline zeroindex(A::PtrArray{S,D}) where {S,D} = PtrArray(zstridedpointer(A), size(A), Val{D}())

# @inline function PtrArray(ptr::StridedPointer

@generated function calc_strides_len(::Type{T}, s::Tuple{Vararg{StaticInt,N}}) where {T, N}
    L = sizeof(T)
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, static_expr(L))
        L *= s.parameters[n].parameters[1]
    end
    Expr(:tuple, t, static_expr(L))
end
@generated function calc_strides_len(::Type{T}, s::Tuple{Vararg{Any,N}}) where {T, N}
    last_sx = :s_0
    q = Expr(:block, Expr(:meta,:inline), Expr(:(=), last_sx, static_expr(sizeof(T))))
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, last_sx)
        new_sx = Symbol(:s_,n)
        push!(q.args, Expr(:(=), new_sx, Expr(:call, :vmul_fast, last_sx, Expr(:ref, :s, n))))
        last_sx = new_sx
    end
    push!(q.args, Expr(:tuple, t, last_sx))
    q
end

# @inlines are because we want to make sure the compiler has the chance to avoid the allocation
# @inline function FixedSizeArray{T}(::UndefInitializer, s::Tuple{Vararg{StaticInt,N}}) where {N,T}
#     x, L = calc_strides_len(s)
#     b = MemoryBuffer{T}(undef, L)
#     ptr = VectorizationBase.align(pointer(r))
#     FixedSizeArray(ptr, s, x, b)
# end
# @inline function FixedSizeArray(ptr::Ptr{T}, s::S, x::X, b::NTuple)
#     FixedSizeArray(PtrArray(ptr, s, x), b)
# end

@inline function StrideArray{T}(::UndefInitializer, s::Tuple{Vararg{Integer,N}}) where {N,T}
    x, L = calc_strides_len(T,s)
    b = undef_memory_buffer(T, L ÷ static_sizeof(T))
    # For now, just trust Julia's alignment heuristics are doing the right thing
    # to save us from having to over-allocate
    # ptr = VectorizationBase.align(pointer(b))
    ptr = pointer(b)
    StrideArray(ptr, s, x, b, all_dense(Val{N}()))
end
@inline function StrideArray(ptr::Ptr{T}, s::S, x::X, b, ::Val{D}) where {S,X,T,D}
    StrideArray(PtrArray(ptr, s, x, Val{D}()), b)
end

@inline PtrArray(A::StrideArray) = A.ptr



@generated rank_to_sortperm_val(::Val{R}) where {R} = :(Val{$(rank_to_sortperm(R))}())
@inline function similar_layout(A::AbstractStrideArray{S,D,T,N,C,B,R}) where {S,D,T,N,C,B,R}
    permutedims(similar(permutedims(A, rank_to_sortperm_val(Val{R}()))), Val{R}())
end
@inline function similar_layout(A::AbstractArray)
    b = preserve_buffer(A)
    GC.@preserve b begin
        similar_layout(PtrArray(A))
    end
end
@inline function Base.similar(A::AbstractStrideArray{S,D,T}) where {S,D,T}
    StrideArray{T}(undef, size(A))
end


@inline function Base.view(A::StrideArray, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K}
    StrideArray(view(A.ptr, i...), A.data)
end
@inline function zview(A::StrideArray, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K}
    StrideArray(zview(A.ptr, i...), A.data)
end
@inline function Base.permutedims(A::StrideArray, ::Val{P}) where {P}
    StrideArray(permutedims(A.ptr, Val{P}()), A.data)
end
@inline Base.adjoint(a::StrideVector) = StrideArray(adjoint(a.ptr), a.data)
# function calc_padding(nrow::Int, T)
#     W = VectorizationBase.pick_vector_width(T)
#     W > nrow ? VectorizationBase.nextpow2(nrow) : VectorizationBase.align(nrow, T)
# end



