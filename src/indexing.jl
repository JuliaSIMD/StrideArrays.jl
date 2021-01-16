



@noinline ThrowBoundsError(A, i) = (println("A of length $(length(A))."); throw(BoundsError(A, i)))
                                
Base.IndexStyle(::Type{<:AbstractStrideArray}) = IndexCartesian()
Base.IndexStyle(::Type{<:AbstractStrideVector{<:Any,<:Any,<:Any,1}}) = IndexLinear()
@generated function Base.IndexStyle(::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}}) where {S,D,T,N,C,B,R}
    # if is column major || is a transposed contiguous vector
    if all(D) && ((isone(C) && R === ntuple(identity, Val(N))) || (C === 2 && R === (2,1) && S <: Tuple{One,Integer}))
        :(IndexLinear())
    else
        :(IndexCartesian())
    end          
end

Base.@propagate_inbounds Base.getindex(A::AbstractStrideVector, i::Int, j::Int) = A[i]
@inline function Base.getindex(A::PtrArray{S,D,T,K}, i::Vararg{Integer,K}) where {S,D,T,K}
    @boundscheck checkbounds(A, i...)
    vload(stridedpointer(A), i)
end
@inline function Base.getindex(A::AbstractStrideArray{S,D,T,K}, i::Vararg{Integer,K}) where {S,D,T,K}
    b = preserve_buffer(A)
    P = PtrArray(A)
    GC.@preserve b begin
        @boundscheck checkbounds(P, i...)
        vload(stridedpointer(P), i)
    end
end
@inline function Base.setindex!(A::PtrArray{S,D,T,K}, v, i::Vararg{Integer,K}) where {S,D,T,K}
    @boundscheck checkbounds(A, i...)
    vstore!(stridedpointer(A), v, i)
    v
end
@inline function Base.setindex!(A::AbstractStrideArray{S,D,T,K}, v, i::Vararg{Integer,K}) where {S,D,T,K}
    b = preserve_buffer(A)
    P = PtrArray(A)
    GC.@preserve b begin
        @boundscheck checkbounds(P, i...)
        vstore!(stridedpointer(P), v, i)
    end
    v
end
@inline function Base.getindex(A::PtrArray, i::Integer)
    @boundscheck checkbounds(A, i)
    vload(stridedpointer(A), (i - one(i),))
end
@inline function Base.getindex(A::AbstractStrideArray, i::Integer)
    b = preserve_buffer(A)
    P = PtrArray(A)
    GC.@preserve b begin
        @boundscheck checkbounds(P, i)
        vload(stridedpointer(P), (i - one(i),))
    end
end
@inline function Base.setindex!(A::PtrArray, v, i::Integer)
    @boundscheck checkbounds(A, i)
    vstore!(stridedpointer(A), v, (i - one(i),))
    v
end
@inline function Base.setindex!(A::AbstractStrideArray, v, i::Integer)
    b = preserve_buffer(A)
    P = PtrArray(A)
    GC.@preserve b begin
        @boundscheck checkbounds(P, i)
        vstore!(stridedpointer(P), v, (i - one(i),))
    end
    v
end

