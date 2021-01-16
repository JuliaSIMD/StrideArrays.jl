
@inline VectorizationBase.stridedpointer(A::PtrArray) = A.ptr
@inline VectorizationBase.stridedpointer(A::StrideArray) = A.ptr.ptr

ArrayInterface.device(::AbstractStrideArray) = ArrayInterface.CPUPointer()

# AbstractStrideArray{S,D,T,N,C,B,R,X,O}

# contiguous_axis(A), contiguous_batch_size(A), stride_rank(A), bytestrides(A), offsets(A))


ArrayInterface.contiguous_axis(::Type{<:AbstractStrideArray{S,D,T,N,C}}) where {S,D,T,N,C} = ArrayInterface.Contiguous{C}()
ArrayInterface.contiguous_batch_size(::Type{<:AbstractStrideArray{S,D,T,N,C,B}}) where {S,D,T,N,C,B} = ArrayInterface.ContiguousBatch{B}()
ArrayInterface.stride_rank(::Type{<:AbstractStrideArray{S,D,T,N,C,B,R}}) where {S,D,T,N,C,B,R} = ArrayInterface.StrideRank{R}()
ArrayInterface.dense_dims(::Type{<:AbstractStrideArray{S,D}}) where {S,D} = DenseDims{D}()

