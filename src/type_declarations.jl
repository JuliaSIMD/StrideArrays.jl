


@inline undef_memory_buffer(::Type{T}, ::StaticInt{L}) where {T,L} = MemoryBuffer{L,T}(undef)
@inline undef_memory_buffer(::Type{T}, L) where {T} = Vector{T}(undef, L)

struct StrideArray{S,D,T,N,C,B,R,X,O,A} <: AbstractStrideArray{S,D,T,N,C,B,R,X,O}
    ptr::PtrArray{S,D,T,N,C,B,R,X,O}
    data::A
end

@inline VectorizationBase.stridedpointer(A::StrideArray) = A.ptr.ptr

const StrideVector{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,1,C,B,R,X,O,A}
const StrideMatrix{S,D,T,C,B,R,X,O,A} = StrideArray{S,D,T,2,C,B,R,X,O,A}

@inline StrideArray(A::AbstractArray) = StrideArray(PtrArray(A), A)

@inline VectorizationBase.preserve_buffer(A::MemoryBuffer) = A
@inline VectorizationBase.preserve_buffer(A::StrideArray) = LoopVectorization.preserve_buffer(A.data)

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

