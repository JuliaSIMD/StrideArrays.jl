


@inline undef_memory_buffer(::Type{T}, ::StaticInt{L}) where {T,L} = MemoryBuffer{L,T}(undef)
@inline undef_memory_buffer(::Type{T}, L) where {T} = Vector{T}(undef, L)

abstract type AbstractStrideArray{S,D,T<:VectorizationBase.NativeTypes,N,C,B,R,X,O} <: DenseArray{T,N} end
abstract type AbstractPtrStrideArray{S,D,T,N,C,B,R,X,O} <: AbstractStrideArray{S,D,T,N,C,B,R,X,O} end

struct PtrArray{S,D,T,N,C,B,R,X,O} <: AbstractPtrStrideArray{S,D,T,N,C,B,R,X,O}
    ptr::StridedPointer{T,N,C,B,R,X,O}
    size::S
end
@inline function PtrArray(ptr::StridedPointer{T,N,C,B,R,X,O}, size::S, ::Val{D}) where {S,D,T,N,C,B,R,X,O}
    PtrArray{S,D,T,N,C,B,R,X,O}(ptr, size)
end

struct StrideArray{S,D,T,N,C,B,R,X,O,A} <: AbstractStrideArray{S,D,T,N,C,B,R,X,O}
    ptr::PtrArray{S,D,T,N,C,B,R,X,O}
    data::A
end

const StrideVector{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,1,C,B,R,X,O,A}
const StrideMatrix{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,2,C,B,R,X,O,A}
const AbstractStrideVector{S,D,T,C,B,R,X,O} = AbstractStrideArray{S,D,T,1,C,B,R,X,O}
const AbstractStrideMatrix{S,D,T,C,B,R,X,O} = AbstractStrideArray{S,D,T,2,C,B,R,X,O}

@inline StrideArray(A::AbstractArray) = StrideArray(PtrArray(A), A)


@inline LoopVectorization.preserve_buffer(A::MemoryBuffer) = A
@inline LoopVectorization.preserve_buffer(A::StrideArray) = LoopVectorization.preserve_buffer(A.data)
@inline LoopVectorization.preserve_buffer(A::PtrArray) = nothing

@inline maybe_ptr_array(A) = A
@inline maybe_ptr_array(A::AbstractArray) = maybe_ptr_array(ArrayInterface.device(A), A)
@inline maybe_ptr_array(::ArrayInterface.CPUPointer, A::AbstractArray) = PtrArray(A)
@inline maybe_ptr_array(_, A::AbstractArray) = A



function gc_preserve_call(ex, skip=0)
    q = Expr(:block)
    call = Expr(:call, esc(ex.args[1]))
    gcp = Expr(:gc_preserve, call)
    for i ∈ 2:length(ex.args)
        arg = ex.args[i]
        if i+1 ≤ skip
            push!(call.args, arg)
            continue
        end
        A = gensym(:A); buffer = gensym(:buffer);
        if arg isa Expr && arg.head === :kw
            push!(call.args, Expr(:kw, arg.args[1], Expr(:call, :maybe_ptr_array, A)))
            arg = arg.args[2]
        else
            push!(call.args, Expr(:call, :maybe_ptr_array, A))
        end
        push!(q.args, :($A = $(esc(arg))))
        push!(q.args, Expr(:(=), buffer, Expr(:call, :preserve_buffer, A)))
        push!(gcp.args, buffer)
    end
    push!(q.args, gcp)
    q
end
"""
  @gc_preserve foo(A, B, C)

Apply to a single, non-nested, function call. It will `GC.@preserve` all the arguments, and substitute suitable arrays with `PtrArray`s.
This has the benefit of potentially allowing statically sized mutable arrays to be both stack allocated, and passed through a non-inlined function boundary.
"""
macro gc_preserve(ex)
    @assert ex.head === :call
    gc_preserve_call(ex)
end


# const FVector{L,T} = StrideVector{Tuple{StaticInt{L}},(true,),T,1,0,(1,)}

# @inline function StrideArray{S,T,N,C,B,R,X,O,D}(ptr, sz, sx, data) where {S,T,N,C,B,R,X,O,D}
    # StrideArray{S,T,N,C,B,R,X,O,D}(PtrArray{S,T,N,X,SN,XN,V}(ptr, sz, sx), data)
# end
# struct FixedSizeArray{S,D,T,N,C,B,R,X,O} <: AbstractStrideArray{S,D,T,N,C,B,R,X,O}
#     ptr::PtrArray{S,D,T,N,C,B,R,X,O}
#     data::MemoryBuffer{L,T}
# end

# struct ConstantArray{S,T,N,X,L} <: AbstractStrideArray{S,T,N,X,0,0,false}
#     data::NTuple{L,Core.VecElement{T}}
# end
# @generated function check_N(::Val{N}, ::Type{S}, ::Type{X}) where {N,S,X}
#     if N == length(S.parameters) == length(X.parameters)
#         nothing
#     else
#         throw("Dimensions declared: $N; Size: $S; Strides: $X")
#     end
# end
# const LazyArray{F,S,T,N,X,SN,XN,V,L} = VectorizationBase.LazyMap{F,A,A<:AbstractStrideArray{S,T,N,X,SN,XN,V,L}}
# struct LazyMap{F,S,T,N,X,SN,XN,V,A<:AbstractStrideArray{S,T,N,X,SN,XN,V}} <: AbstractStrideArray{S,T,N,X,SN,XN,V}
#     f::F
#     ptr::A
# end

# @inline size_tuple(A::PtrArray) = A.size
# @inline stride_tuple(A::PtrArray) = A.stride
# @inline size_tuple(A::AbstractStrideArray) = A.ptr.size
# @inline stride_tuple(A::AbstractStrideArray) = A.ptr.stride


# const AbstractStrideVector{M,T,X1,SN,XN,V} = AbstractStrideArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
# const AbstractStrideMatrix{M,N,T,X1,X2,SN,XN,V} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
# const StrideVector{M,T,X1,SN,XN,V} = StrideArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
# const StrideMatrix{M,N,T,X1,X2,SN,XN,V} = StrideArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
# const FixedSizeVector{M,T,X1,L} = FixedSizeArray{Tuple{M},T,1,Tuple{X1},L}
# const FixedSizeMatrix{M,N,T,X1,X2,L} = FixedSizeArray{Tuple{M,N},T,2,Tuple{X1,X2},L}
# const ConstantVector{M,T,X1,L} = ConstantArray{Tuple{M},T,1,Tuple{X1},L}
# const ConstantMatrix{M,N,T,X1,X2,L} = ConstantArray{Tuple{M,N},T,2,Tuple{X1,X2},L}
# const PtrVector{M,T,X1,SN,XN,V} = PtrArray{Tuple{M},T,1,Tuple{X1},SN,XN,V}
# const PtrMatrix{M,N,T,X1,X2,SN,XN,V} = PtrArray{Tuple{M,N},T,2,Tuple{X1,X2},SN,XN,V}
# const AbstractFixedSizeArray{S,T,N,X,V} = AbstractStrideArray{S,T,N,X,0,0,V}
# const AbstractFixedSizeVector{S,T,X,V} = AbstractStrideArray{Tuple{S},T,1,Tuple{X},0,0,V}
# const AbstractFixedSizeMatrix{M,N,T,X1,X2,V} = AbstractStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V}
# const AbstractMutableFixedSizeArray{S,T,N,X,V} = AbstractMutableStrideArray{S,T,N,X,0,0,V}
# const AbstractMutableFixedSizeVector{S,T,X,V} = AbstractMutableStrideArray{Tuple{S},T,1,Tuple{X},0,0,V}
# const AbstractMutableFixedSizeMatrix{M,N,T,X1,X2,V} = AbstractMutableStrideArray{Tuple{M,N},T,2,Tuple{X1,X2},0,0,V}


@inline Base.pointer(A::StrideArray) = A.ptr.ptr.p
# @inline Base.pointer(A::FixedSizeArray) = A.ptr.ptr
@inline Base.pointer(A::PtrArray) = A.ptr.p

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::StrideArray{S,D,T}) where {S,D,T} = A.ptr.ptr.p
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T}) where {S,T} = A.ptr.ptr.p
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::PtrArray{S,D,T}) where {S,D,T} = A.ptr.p

@inline Base.elsize(::AbstractStrideArray{<:Any,<:Any,T}) where {T} = sizeof(T)



# @inline Base.pointer(A::FixedSizeArray{S,T,N,X,L,Ptr{T}}) where {S,T,N,X,L} = A.ptr
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T,N,X,L,Ptr{T}}) where {S,T,N,X,L} = A.ptr
# @inline Base.pointer(A::FixedSizeArray{S,T,N,X,L,Nothing}) where {S,T,N,X,L} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::FixedSizeArray{S,T,N,X,L,Nothing}) where {S,T,N,X,L} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(A))
# @inline Base.pointer(A::LazyMap) = pointer(A.ptr)
# @inline Base.unsafe_convert(::Type{Ptr{T}}, A::LazyMap{S,T}) where {S,T} = pointer(A.ptr)



