
@inline ArrayInterface.size(A::PtrArray) = A.size
@inline ArrayInterface.size(A::StrideArray) = A.ptr.size
@inline VectorizationBase.bytestrides(A::PtrArray) = A.ptr.strd
@inline VectorizationBase.bytestrides(A::StrideArray) = A.ptr.ptr.strd

@inline bytestride(A, n) = VectorizationBase.bytestrides(A)[n]


@generated function ArrayInterface.strides(A::PtrArray{S,D,T,N}) where {S,D,T,N}
    shifter = static_expr(VectorizationBase.intlog2(sizeof(T)))
    x = Expr(:tuple)
    for n in 1:N
        push!(x.args, Expr(:call, :(>>>), Expr(:ref, :x, n), shifter))
    end
    quote
        $(Expr(:meta,:inline))
        x = A.ptr.strd
        $x
    end
end
@inline ArrayInterface.strides(A::StrideArray) = strides(A.ptr)

@inline Base.size(A::AbstractStrideArray) = map(Int, ArrayInterface.size(A))
@inline Base.strides(A::AbstractStrideArray) = map(Int, ArrayInterface.strides(A))

@inline create_axis(s, ::Zero) = CloseOpen(s)
@inline create_axis(s, ::One) = One():s
@inline create_axis(s, o) = CloseOpen(s, s+o)

@inline ArrayInterface.axes(A::AbstractStrideArray) = map(create_axis, size(A), offsets(A))
@inline Base.axes(A::AbstractStrideArray) = axes(A)

@inline ArrayInterface.offsets(A::PtrArray) = A.ptr.offsets
@inline ArrayInterface.offsets(A::StrideArray) = A.ptr.ptr.offsets

@inline ArrayInterface.static_length(A::AbstractStrideArray) = prod(size(A))

# type stable, because index known at compile time
@inline type_stable_select(t::NTuple, ::StaticInt{N}) where {N} = t[N]
@inline type_stable_select(t::Tuple, ::StaticInt{N}) where {N} = t[N]
# type stable, because tuple is homogenous
@inline type_stable_select(t::NTuple, i::Integer) = t[i]
# make the tuple homogenous before indexing
@inline type_stable_select(t::Tuple, i::Integer) = map(Int, t)[i]

@inline function Base.axes(A::AbstractStrideVector, i::Integer)
    if i == 1
        o = type_stable_select(offsets(A), i)
        s = type_stable_select(size(A), i)
        return create_axis(s, o)
    else
        return One():1
    end
end
@inline function Base.axes(A::AbstractStrideArray, i::Integer)
    o = type_stable_select(offsets(A), i)
    s = type_stable_select(size(A), i)
    create_axis(s, o)
end

@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{One}) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::Base.OneTo) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::AbstractUnitRange) = Zero():(last(r)-first(r))

@inline zeroindex(r::CloseOpen{Zero}) = r
@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{Zero}) = r

