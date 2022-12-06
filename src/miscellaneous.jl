
function gc_preserve_call_expr(c, K)
  q = Expr(:block, Expr(:meta, :inline))
  for k ∈ 1:K
    push!(c.args, :(@inbounds(args[$k])))
  end
  push!(q.args, gc_preserve_call(c, 1))
  push!(q.args, :A)
  q
end

# TODO: replace this with `LoopVectorization.gc_preserve_vmap!` after making that version switch to `stridedpointers`+ size info
@generated function gc_preserve_map!(
  f::F,
  A::AbstractStrideArray,
  args::Vararg{AbstractArray,K},
) where {F,K}
  gc_preserve_call_expr(:(vmap!(f, A)), K)
end

@inline Base.map!(
  f::F,
  A::AbstractStrideArray,
  arg1::AbstractArray,
  args::Vararg{AbstractArray,K},
) where {F,K} = gc_preserve_map!(f, A, arg1, args...)
# these two definitions are to avoid ambiguities
@inline Base.map!(f::F, A::AbstractStrideArray, arg::AbstractArray) where {F} =
  gc_preserve_map!(f, A, arg)
@inline Base.map!(
  f::F,
  A::AbstractStrideArray,
  arg1::AbstractArray,
  arg2::AbstractArray,
) where {F} = gc_preserve_map!(f, A, arg1, arg2)
@inline Base.map(f::F, A::AbstractStrideArray, args::Vararg{Any,K}) where {F,K} =
  gc_preserve_map!(f, A, args...)
using StaticArraysCore: StaticArray, SArray, MArray
@inline Base.map(f::F, A::AbstractStrideArray, B::StaticArray, args::Vararg{AbstractArray,K}) where {F,K} =
  gc_preserve_map!(f, A, B, args...)
@inline function Base.map(f::F, A::AbstractStrideArray, B::SArray, args::Vararg{AbstractArray,K}) where {F,K}
  BM = MArray(B)
  gc_preserve_map!(f, A, BM, args...)
end
@inline Base.reduce(op::O, A::AbstractStrideArray{<:Number}) where {O} =
  @gc_preserve vreduce(op, A)
@inline Base.reduce(::typeof(vcat), A::AbstractStrideArray{<:Number}) = A
@inline Base.reduce(::typeof(hcat), A::AbstractStrideArray{<:Number}) = A
@inline function gc_preserve_mapreduce(
  f::F,
  op::O,
  A::AbstractStrideArray,
  args::Vararg{AbstractArray,K},
) where {F,O,K}
  gc_preserve_call_expr(:(vmapreduce(f, op, A)), K)
end
@inline Base.mapreduce(
  f::F,
  op::O,
  A::AbstractStrideArray,
  args::Vararg{AbstractArray,K},
) where {F,O,K} = gc_preserve_mapreduce(f, op, A, args...)
@inline function Base.mapreduce(f::F, op::O, A::AbstractStrideArray) where {F,O}
  if (LoopVectorization.check_args(A) && LoopVectorization.all_dense(A))
    @gc_preserve vmapreduce(f, op, A)
  else
    return @gc_preserve Base.invoke(
      Base.mapreduce,
      Tuple{F,O,AbstractArray{eltype(A)}},
      f,
      op,
      A,
    )
  end
end

import Statistics, VectorizedStatistics

for (op, r) ∈ ((:max, :maximum), (:min, :minimum))
  vr = Symbol('v', r)
  @eval begin
    @inline function Base.reduce(::typeof($op), A::AbstractStrideArray{<:Number}; dim = (:), dims = (:))
      @gc_preserve VectorizedStatistics.$vr(A; dim, dims)
    end
    @inline function Base.$r(A::AbstractStrideArray; dim=(:), dims = (:))
      @gc_preserve VectorizedStatistics.$vr(A; dim, dims)
    end
  end
end


@inline Base.reduce(::typeof(+), A::AbstractStrideArray{<:Number}; dim = :, dims = :, multithreaded = False()) =
  @gc_preserve VectorizedStatistics.vsum(A; dim, dims, multithreaded)
@inline Base.sum(A::AbstractStrideArray; dim=:, dims = :, multithreaded = False()) =
  @gc_preserve VectorizedStatistics.vsum(A; dim, dims, multithreaded)

@inline Statistics.mean(A::AbstractStrideArray; dim=:, dims = :, multithreaded = False()) =
  @gc_preserve VectorizedStatistics.vmean(A; dim, dims, multithreaded)
@inline Statistics.std(A::AbstractStrideArray; dim=:, dims = :, mean=nothing, corrected=true, multithreaded = False()) =
  @gc_preserve VectorizedStatistics.vstd(A; dim, dims, mean, corrected, multithreaded)
@inline Statistics.var(A::AbstractStrideArray; dim=:, dims = :, mean=nothing, corrected=true, multithreaded = False()) =
  @gc_preserve VectorizedStatistics.vvar(A; dim, dims, mean, corrected, multithreaded)

for f = (:cov, :cor)
  vf = Symbol('v', f)
  @eval begin
    @inline Statistics.$f(x::AbstractStrideVector, y::AbstractStrideVector; corrected = true, multithreaded=False()) = @gc_preserve VectorizedStatistics.$vf(x, y, corrected, multithreaded)
    @inline Statistics.$f(x::AbstractStrideMatrix; dims::Int=1, corrected::Bool=true,multithreaded=False()) = @gc_preserve VectorizedStatistics.$vf(x, dims, corrected, multithreaded)
  end
end

function Base.copyto!(
  B::AbstractStrideArray{<:Any,N},
  A::AbstractStrideArray{<:Any,N},
) where {N}
  @turbo for I ∈ eachindex(A, B)
    B[I] = A[I]
  end
  B
end

# why not `vmapreduce`?
@inline function Base.maximum(::typeof(abs), A::AbstractStrideArray{T}) where {T}
  s = typemin(T)
  @turbo for i ∈ eachindex(A)
    s = max(s, abs(A[i]))
  end
  s
end

@inline function Base.vcat(A::AbstractStrideMatrix, B::AbstractStrideMatrix)
  MA, NA = size(A)
  MB, NB = size(B)
  @assert NA == NB
  TC = promote_type(eltype(A), eltype(B))
  C = StrideArray{TC}(undef, (MA + MB, NA))
  # TODO: Actually handle offsets
  @assert offsets(A) === offsets(B)
  @assert offsets(A) === offsets(C)
  @turbo for j ∈ axes(A, 2), i ∈ axes(A, 1)
    C[i, j] = A[i, j]
  end
  @turbo for j ∈ axes(B, 2), i ∈ axes(B, 1)
    C[i+MA, j] = B[i, j]
  end
  C
end


@inline function make_stride_dynamic(p::StridedPointer{T,N,C,B,R}) where {T,N,C,B,R}
  si = StrideIndex{N,R,C}(map(Int, strides(p)), offsets(p))
  stridedpointer(pointer(p), si, StaticInt{B}())
end
@inline function make_dynamic(A::PtrArray)
  PtrArray(make_stride_dynamic(stridedpointer(A)), Base.size(A), val_dense_dims(A))
end
@inline function make_dynamic(A::StrideArray)
  StrideArray(make_dynamic(PtrArray(A)), A.data)
end

@inline function Base.exp(A::AbstractStrideArray)
  B = copy(A)
  GC.@preserve C = LinearAlgebra.exp!(B)
  B === C || copyto!(B, C)
  return B
end
