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
    # @show expr.args
    N = length(expr.args)
    n = 2
    randtypes = (:Float32,:Float64,:Int,:Int32,:Int64,:UInt,:UInt32,:UInt64)
    arg2 = expr.args[2]
    if arg2 ∈ randtypes
        T = arg2
        n += 1
    elseif arg2 isa Expr && arg2.head === :($)
        T = esc(first(arg2.args))
        n += 1
    else
        T = Float64
    end
    s = Expr(:tuple)
    for i in n:N
        aᵢ = expr.args[i]
        if aᵢ isa Integer
            push!(s.args, static_expr(Int(aᵢ)))
        else
            push!(s.args, esc(aᵢ))
        end
    end
    f! = Symbol(expr.args[1], :!)
    array = :(StrideArray{$T}(undef, $s))
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

macro StrideArray(expr, args...)
    rand_expr(expr, args...)
end

