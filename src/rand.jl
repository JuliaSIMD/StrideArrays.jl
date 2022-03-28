using VectorizedRNG: local_rng

@inline function Random.rand!(A::AbstractStrideArray, args::Vararg{Any,K}) where {K}
    rand!(local_rng(), A, args...)
end
@inline function Random.randn!(A::AbstractStrideArray, args::Vararg{Any,K}) where {K}
    randn!(local_rng(), A, args...)
end
## ignore type...
@inline Random.rand!(A::AbstractStrideArray, ::Type{T}) where {T} = rand!(local_rng(), A)


function rand_expr(expr, args...)
    array = :(StrideArray(undef))
    for i in 2:length(expr.args)
        aᵢ = expr.args[i]
        if aᵢ isa Integer
            push!(array.args, StaticInt(Int(aᵢ)))
        elseif Meta.isexpr(aᵢ, :$, 1)
            push!(array.args, Expr(:call, GlobalRef(StrideArrays, :StaticInt), (only(aᵢ.args))))
        else
            push!(array.args, esc(aᵢ))
        end
    end
    f! = Symbol(expr.args[1], :!)
    # we call PtrArray so that `rand!` doesn't force heap allocation.
    call = Expr(:call, f!, Expr(:call, :PtrArray, :array))
    for arg ∈ args
        push!(call.args, esc(arg))
    end
    quote
        array = $array
        b = array.data
        # we add GC.@preserve to save the array.
        GC.@preserve b $call
        array
    end
end

"""
    @StrideArray rand(Float32, 3, 4)
    @StrideArray randn($(7>>1),  4)
    @StrideArray rand(7>>1,  4)

Creates a random `StrideArray`.
The default element type is `Float64`.
Dimensions will be statically sized if specified by an integer literal, or if interpolated.
"""
macro StrideArray(expr, args...)
    rand_expr(expr, args...)
end

