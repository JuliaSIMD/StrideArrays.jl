
@inline VectorizationBase.stridedpointer(A::PtrArray) = A.ptr
@inline VectorizationBase.stridedpointer(A::StrideArray) = A.ptr.ptr

ArrayInterface.device(::AbstractStrideArray) = ArrayInterface.CPUPointer()

# AbstractStrideArray{S,D,T,N,C,B,R,X,O}

# contiguous_axis(A), contiguous_batch_size(A), stride_rank(A), bytestrides(A), offsets(A))


ArrayInterface.contiguous_axis(::Type{<:AbstractStrideArray{S,D,T,N,C}}) where {S,D,T,N,C} = StaticInt{C}()
ArrayInterface.contiguous_batch_size(::Type{<:AbstractStrideArray{S,D,T,N,C,B}}) where {S,D,T,N,C,B} = ArrayInterface.StaticInt{B}()

@generated function ArrayInterface.stride_rank(::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}}) where {S,D,T,N,C,B,R}
    t = Expr(:tuple)
    for r ∈ R
        push!(t.args, static_expr(r::Int))
    end
    t
end
@generated function ArrayInterface.dense_dims(::Type{<:AbstractStrideArray{S,D}}) where {S,D}
    t = Expr(:tuple)
    for d ∈ D
        push!(t.args, static_expr(d::Bool))
    end
    t
end

