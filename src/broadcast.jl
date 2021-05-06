
using LoopVectorization: forbroadcast

_extract(::Type{StaticInt{N}}) where {N} = N::Int
_extract(_) = nothing

abstract type AbstractStrideStyle{S,N} <: Base.Broadcast.AbstractArrayStyle{N} end
struct LinearStyle{S,N,R} <: AbstractStrideStyle{S,N} end
struct CartesianStyle{S,N} <: AbstractStrideStyle{S,N} end
Base.BroadcastStyle(::Type{A}) where {S,D,T,N,C,B,R,A<:AbstractStrideArray{S,D,T,N,C,B,R}} = all(D) ? LinearStyle{S,N,R}() : CartesianStyle{S,N}()
# Base.BroadcastStyle(::Type{A}) where {S,T,N,X,SN,XN,A<:AbstractStrideArray{S,T,N,X,SN,XN,true}} = CartesianStyle{S,N}()
# function reverse_simplevec(S, N = length(S))
#     Srev = Expr(:curly, :Tuple)
#     for n ∈ 1:N
#         push!(Srev.args, S.parameters[N + 1 - n])
#     end
#     if N == 1
#         N += 1
#         insert!(Srev.args, 2, 1)
#     end
#     Srev, N
# # end
# @generated function Base.BroadcastStyle(::Type{Adjoint{T,A}}) where {S,T,N,A<:AbstractStrideArray{S,T,N}}
#     Srev, Nrev = reverse_simplevec(S, N)
#     Expr(:call, Expr(:curly, :CartesianStyle, Srev, Nrev))
# end
# @generated function Base.BroadcastStyle(::Type{Transpose{T,A}}) where {S,T,N,A<:AbstractStrideArray{S,T,N}}
#     Srev, Nrev = reverse_simplevec(S, N)
#     Expr(:call, Expr(:curly, :CartesianStyle, Srev, Nrev))
# end

const StrideArrayProduct = Union{
    LoopVectorization.Product{<:AbstractStrideArray},
    LoopVectorization.Product{<:Any,<:AbstractStrideArray},
    LoopVectorization.Product{<:AbstractStrideArray,<:AbstractStrideArray}
    # LoopVectorization.Product{Adjoint{<:Any,<:AbstractFixedSizeArray}},
    # LoopVectorization.Product{Transpose{<:Any,<:AbstractFixedSizeArray}},
    # LoopVectorization.Product{<:Any,Adjoint{<:Any,<:AbstractFixedSizeArray}},
    # LoopVectorization.Product{<:Any,Transpose{<:Any,<:AbstractFixedSizeArray}}
}


@generated function Base.BroadcastStyle(::Type{P}) where {SA,A<:AbstractStrideArray{SA}, SB, B<:AbstractStrideArray{SB}, P<:LoopVectorization.Product{A,B}}
    t = Expr(:curly, :Tuple)
    M = _extract(SA.parameters[1])
    if M === nothing
        push!(t.args, :Int)
    else
        push!(t.args, Expr(:curly, :StaticInt, M))
    end
    if isone(length(SB.parameters))
        return :(CartesianStyle{$t,1}())
    else
        N = _extract(SB.parameters[2])
        if N === nothing
            push!(t.args, :Int)
        else
            push!(t.args, Expr(:curly, :StaticInt, N))
        end
    end
    :(CartesianStyle{$t,2}())
end

@generated Base.BroadcastStyle(a::CartesianStyle{S,N1}, b::Base.Broadcast.DefaultArrayStyle{N2}) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : :a
@generated Base.BroadcastStyle(a::LinearStyle{S,N1}, b::Base.Broadcast.DefaultArrayStyle{N2}) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : CartesianStyle{S,N1}()
Base.BroadcastStyle(a::CartesianStyle{S,N}, b::AbstractStrideStyle{S,N}) where {S,N} = a
Base.BroadcastStyle(a::LinearStyle{S,N,R}, b::LinearStyle{S,N,R}) where {S,N,R} = a # ranks match
Base.BroadcastStyle(a::LinearStyle{S,N}, b::LinearStyle{S,N}) where {S,N} = CartesianStyle{S,N}() # ranks don't match
@generated function Base.BroadcastStyle(a::AbstractStrideStyle{S1,N1}, b::AbstractStrideStyle{S2,N2}) where {S1,S2,N1,N2}
#    @show N2, N1
    N2 > N1 && return :(Base.Broadcast.Unknown())
    S = Expr(:curly, :Tuple)
    # foundfirstdiff = false
    for n ∈ 1:N2#min(N1,N2)
        _s1 = _extract(S1.parameters[n])
        _s2 = _extract(S2.parameters[n])
        s1 = (_s1 === nothing ? -1 : _s1)::Int
        s2 = (_s2 === nothing ? -1 : _s2)::Int
        if s1 == s2
            push!(S.args, s1)
        elseif s2 == 1
            # foundfirstdiff = true
            push!(S.args, s1)
        elseif s1 == 1
            # foundfirstdiff || return Base.Broadcast.Unknown()
            # foundfirstdiff = true
            push!(S.args, s2)
        elseif s2 == -1
            push!(S.args, s1)
        elseif s1 == -1
            push!(S.args, s2)
        else
            throw("Mismatched sizes: $S1, $S2.")
        end
    end
    # if N2 > N1
    #     for n ∈ N1+1:N2
    #         push!(S.args, S2.parameters[n])
    #     end
    # else
    # if N1 > N2
    for n ∈ N2+1:N1
        push!(S.args, S1.parameters[n])
    end
    # end
    Expr(:call, Expr(:curly, :CartesianStyle, S, max(N1,N2)))
end

# function Base.similar(
    # ::Base.Broadcast.Broadcasted{FS}, ::Type{T}
# ) where {S,T<:Union{VectorizationBase.FloatingTypes,StrideArrays.VectorizationBase.IntTypes,StrideArrays.VectorizationBase.UIntTypes},N,FS<:AbstractStrideStyle{S,N}}
@generated function to_tuple(::Type{S}, s) where {N,S<:Tuple{Vararg{Any,N}}}
    t = Expr(:tuple)
    Sp = S.parameters
    for i in 1:length(Sp)
        l = _extract(Sp[i])
        if l === nothing
            push!(t.args, Expr(:ref, :s, i))
        else
            push!(t.args, static_expr(l))
        end
    end
    t
end
function Base.similar(
    bc::Base.Broadcast.Broadcasted{FS}, ::Type{T}
) where {S,T,N,FS<:AbstractStrideStyle{S,N}}
    StrideArray{T}(undef, to_tuple(S,size(bc)))
end
@generated function to_tuple(::Type{S}) where {N,S<:Tuple{Vararg{StaticInt,N}}}
    t = Expr(:tuple)
    Sp = S.parameters
    for i in 1:N
        push!(t.args, static_expr(_extract(Sp[i])))
    end
    t
end
function Base.similar(
    bc::Base.Broadcast.Broadcasted{FS}, ::Type{T}
) where {S<:Tuple{Vararg{StaticInt}},T,N,FS<:AbstractStrideStyle{S,N}}
    StrideArray{T}(undef, to_tuple(S))
end
# Base.similar(sp::StackPointer, ::Base.Broadcast.Broadcasted{FS}, ::Type{T}) where {S,T,FS<:AbstractStrideStyle{S,T},N} = PtrArray{S,T}(sp)


function add_single_element_array!(ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, elementbytes)
    LoopVectorization.pushprepreamble!(ls, Expr(:(=), Symbol("##", destname), Expr(:call, :first, bcname)))
    LoopVectorization.add_constant!(ls, destname, elementbytes)
end
function add_fs_array!(ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, indexes, S, R, C, elementbytes)
    ref = Symbol[]
    # aref = LoopVectorization.ArrayReference(bcname, ref)
    vptrbc = LoopVectorization.vptr(bcname)
    LoopVectorization.add_vptr!(ls, bcname, vptrbc, true) #TODO: is this necessary?
    offset = 0
    Rnew = Int[]
    for (i,n) ∈ enumerate(indexes)
        _s = _extract(S.parameters[i])
        s = (_s === nothing ? -1 : _s)::Int
        # r = R[i]
        # (isone(n) & (stride != 1)) && pushfirst!(ref, LoopVectorization.DISCONTIGUOUS)
        if s == 1
            offset += 1
            bco = bcname
            bcname = Symbol(:_, bcname)
            v = Expr(:call, :view, Expr(:call, :parent, bco))
            foreach(_ -> push!(v.args, :(:)), 1:i - offset)
            push!(v.args, :(One()))
            foreach(_ -> push!(v.args, :(:)), i+1:length(indexes))
            LoopVectorization.pushprepreamble!(ls, Expr(:(=), bcname, v))
        else
            push!(Rnew, R[i])
            push!(ref, loopsyms[n])
        end
    end
    if iszero(length(ref))
        return add_single_element_array!(ls, destname, bcname, elementbytes)
    end
    bctemp = Symbol(:_, bcname)
    mref = LoopVectorization.ArrayReferenceMeta(
        LoopVectorization.ArrayReference(bctemp, ref), fill(true, length(ref)), vptrbc
    )
    sp = sort_indices!(mref, Rnew, C)
    if sp === nothing
        LoopVectorization.pushprepreamble!(ls, Expr(:(=), bctemp,  bcname))
    else
        ssp = Expr(:tuple); append!(ssp.args, sp)
        ssp = Expr(:call, Expr(:curly, :StaticInt, ssp))
        LoopVectorization.pushprepreamble!(ls, Expr(:(=), bctemp,  Expr(:call, :permutedims, bcname, ssp)))
    end
    loadop = LoopVectorization.add_simple_load!(ls, destname, mref, mref.ref.indices, elementbytes)
    LoopVectorization.doaddref!(ls, loadop)
end

# function add_broadcast_adjoint_array!(
#     ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{A}, indices
# ) where {S, T, N, A <: AbstractStrideArray{S,T,N}}
#     if N == 1
#         if _extract(first(S.parameters)) == 1
#             add_single_element_array!(ls, destname, bcname, sizeof(T))
#         else
#             ref = LoopVectorization.ArrayReference(bcname, Symbol[loopsyms[2]])
#             LoopVectorization.add_load!( ls, destname, ref, sizeof(T) )
#         end
#     else
#         add_fs_array!(ls, destname, bcname, loopsyms, indices, S, sizeof(T))
#     end
# end
# function LoopVectorization.add_broadcast!(
#     ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
#     loopsyms::Vector{Symbol}, ::Type{Adjoint{T,A}}, elementbytes::Int = 8
# ) where {T, S, N, A <: AbstractStrideArray{S, T, N}}
#     # @show @__LINE__, A
#     add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, N:-1:1, sizeof(T) )
# end
# function LoopVectorization.add_broadcast!(
#     ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
#     loopsyms::Vector{Symbol}, ::Type{Transpose{T,A}}, elementbytes::Int = 8
# ) where {T, S, N, A <: AbstractStrideArray{S, T, N}}
#     # @show @__LINE__, A
#     add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, N:-1:1, sizeof(T) )
# end
# function LoopVectorization.add_broadcast!(
#     ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
#     loopsyms::Vector{Symbol}, ::Type{PermutedDimsArray{T,N,I1,I2,A}}, elementbytes::Int = 8
# ) where {T, S, N, I1, I2, A <: AbstractStrideArray{S, T, N}}
#     @show @__LINE__, A
#     add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, I2, sizeof(T) )
# end

function sort_indices!(ar, R, C)
    any(i -> R[i-1] ≥ R[i], 2:length(R)) || return nothing
    li = ar.loopedindex; NN = length(li)
    # all(n -> ((Xv[n+1]) % UInt) ≥ ((Xv[n]) % UInt), 1:NN-1) && return nothing    
    inds = LoopVectorization.getindices(ar); offsets = ar.ref.offsets;
    sp = rank_to_sortperm(R)
    # sp = sortperm(reinterpret(UInt,Xv), alg = Base.Sort.DEFAULT_STABLE)
    lib = copy(li); indsb = copy(inds); offsetsb = copy(offsets);
    for i ∈ eachindex(li, inds)
        li[i] = lib[sp[i]]
        inds[i] = indsb[sp[i]]
        offsets[i] = offsetsb[sp[i]]
    end
    C > 0 || pushfirst!(inds, LoopVectorization.DISCONTIGUOUS)
    sp
end

function LoopVectorization.add_broadcast!(
    ls::LoopVectorization.LoopSet, destname::Symbol, bcname::Symbol,
    loopsyms::Vector{Symbol}, ::Type{A}, elementbytes::Int = 8
) where {S,D,T,N,C,B,R, A <: AbstractStrideArray{S,D,T,N,C,B,R}}
    # @show @__LINE__, A
    # Xv = tointvec(X)
    NN = min(N,length(loopsyms))
    op = add_fs_array!(
        ls, destname, bcname, loopsyms, Base.OneTo(NN), S, R, C, sizeof(T)
    )
    op
end

_tuple_type_len(_) = nothing
function _tuple_type_len(::Type{S}) where {N,S<:Tuple{Vararg{StaticInt,N}}}
    L = 1
    for n in 1:N
        L *= _extract(S.parameters[n])::Int
    end
    L
end
# function Base.Broadcast.materialize!(
@generated function _materialize!(
    dest::AbstractStrideArray{S,D,T,N,C,B,R}, bc::BC, ::Val{UNROLL}
) where {S, D, T, N, C, B, R, FS <: LinearStyle{S,N,R}, BC <: Base.Broadcast.Broadcasted{FS}, UNROLL}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n)]
    ls = LoopVectorization.LoopSet(:StrideArrays)
    (inline, u₁, u₂, isbroadcast, W, rs, rc, cls, l1, l2, l3, threads) = UNROLL
    LoopVectorization.set_hw!(ls, rs, rc, cls, l1, l2, l3)
    ls.vector_width = W
    ls.isbroadcast = isbroadcast;
    itersym = first(loopsyms)
    L = _tuple_type_len(S)
    if L === nothing
        Lsym = gensym(:L); Rsym = gensym(:R)
        LoopVectorization.pushprepreamble!(ls, Expr(:(=), Lsym, Expr(:call, :static_length, :dest)))
        LoopVectorization.pushprepreamble!(ls, Expr(:(=), Rsym, Expr(:call, :(:), :(One()), Lsym)))
        LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, Lsym, 1, Rsym, Lsym), itersym)
    else
        LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, L, 1, Symbol(""), Symbol("")), itersym)
    end
    elementbytes = sizeof(T)
    LoopVectorization.add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    storeop = LoopVectorization.add_simple_store!(ls, :dest, LoopVectorization.ArrayReference(:dest, loopsyms), elementbytes)
    LoopVectorization.doaddref!(ls, storeop)
    resize!(ls.loop_order, LoopVectorization.num_loops(ls)) # num_loops may be greater than N, eg Product
    # fallback in case `check_args` fails
    fallback = :(copyto!(dest, Base.Broadcast.instantiate(Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{$N}}(bc.f, bc.args, axes(dest)))))
    Expr(:block, LoopVectorization.setup_call(ls, fallback, LineNumberNode(0), inline, false, u₁, u₂, threads%Int), :dest)
    # ls
end
@generated function _materialize!(
# function _materialize!(
    dest::AbstractStrideArray{S,D,T,N,C,B,R}, bc::BC, ::Val{UNROLL}
) where {S, D, T, N, C, B, R, BC <: Union{Base.Broadcast.Broadcasted,StrideArrayProduct}, UNROLL}
    # 1+1
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n) for n ∈ 1:N]
    ls = LoopVectorization.LoopSet(:StrideArrays)
    (inline, u₁, u₂, isbroadcast, W, rs, rc, cls, l1, l2, l3, threads) = UNROLL
    LoopVectorization.set_hw!(ls, rs, rc, cls, l1, l2, l3)
    ls.vector_width = W
    ls.isbroadcast = isbroadcast;
    destref = LoopVectorization.ArrayReference(:_dest, copy(loopsyms))
    destmref = LoopVectorization.ArrayReferenceMeta(destref, fill(true, length(LoopVectorization.getindices(destref))))
    sp = sort_indices!(destmref, R, C)
    for n ∈ 1:N
        itersym = loopsyms[n]#isnothing(sp) ? n : sp[n]]
        # _s = 
        Sₙ =_extract(S.parameters[n])# (_s === nothing ? -1 : _s)::Int
        if Sₙ === nothing
            Sₙsym = gensym(:Sₙ); Rₙsym = gensym(:Rₙ)
            LoopVectorization.pushprepreamble!(ls, Expr(:(=), Sₙsym, Expr(:call, :size, :dest, n)))
            LoopVectorization.pushprepreamble!(ls, Expr(:(=), Rₙsym, Expr(:call, :(:), :(One()), Sₙsym)))
            LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, Sₙsym, 1, Rₙsym, Sₙsym), itersym)
        else#TODO: handle offsets
            LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, Sₙ::Int, 1, Symbol(""), Symbol("")), itersym)
        end
    end
    elementbytes = sizeof(T)
    # destadj = length(X.parameters) > 1 && last(X.parameters)::Int == 1
    # if destadj
    #     destsym = :dest′
    #     LoopVectorization.pushprepreamble!(ls, Expr(:(=), destsym, Expr(:call, Expr(:(.), :LinearAlgebra, QuoteNode(:Transpose)), :dest)))
    # else
    # end
    LoopVectorization.add_broadcast!(ls, :destination, :bc, loopsyms, BC, elementbytes)
    if isnothing(sp)
        LoopVectorization.pushprepreamble!(ls, Expr(:(=), :_dest, :dest))
    else
        ssp = Expr(:tuple); append!(ssp.args, sp)
        ssp = Expr(:call, Expr(:curly, :StaticInt, ssp))
        LoopVectorization.pushprepreamble!(ls, Expr(:(=), :_dest,  Expr(:call, :permutedims, :dest, ssp)))
    end
    storeop = LoopVectorization.add_simple_store!(ls, :destination, destmref, sizeof(T))
    LoopVectorization.doaddref!(ls, storeop)
    # destref = if destadj
    #     ref = LoopVectorization.ArrayReference(:dest′, reverse(loopsyms))
    # else
    #     if first(X.parameters)::Int != 1
    #         pushfirst!(LoopVectorization.getindices(ref), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    #     end
    #     ref
    # end
    resize!(ls.loop_order, LoopVectorization.num_loops(ls)) # num_loops may be greater than N, eg Product
    # return ls
    # fallback in case `check_args` fails
    fallback = :(copyto!(dest, Base.Broadcast.instantiate(Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{$N}}(bc.f, bc.args, axes(dest)))))
    Expr(:block, LoopVectorization.setup_call(ls, fallback, LineNumberNode(0), inline, false, u₁, u₂, threads%Int), :dest)
end

@inline function Base.Broadcast.materialize!(
    dest::AbstractStrideArray, bc::BC
) where {BC <: Union{Base.Broadcast.Broadcasted,StrideArrayProduct}}
    _materialize!(dest, bc, LoopVectorization.avx_config_val(Val((true,zero(Int8),zero(Int8),true,-1%UInt)), pick_vector_width(eltype(dest))))
end

# @generated function Base.Broadcast.materialize!(
#     dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::BC
# ) where {S, T <: Union{Float32,Float64}, N, X, A <: AbstractStrideArray{S,T,N,X}, BC <: Union{Base.Broadcast.Broadcasted,StrideArrayProduct}}
#     # we have an N dimensional loop.
#     # need to construct the LoopSet
#     loopsyms = [gensym(:n) for n ∈ 1:N]
#     ls = LoopVectorization.LoopSet(:StrideArrays)
#     LoopVectorization.pushprepreamble!(ls, Expr(:(=), :dest, Expr(:call, :parent, :dest′)))
#     for (n,itersym) ∈ enumerate(loopsyms)
#         LoopVectorization.add_loop!(ls, LoopVectorization.Loop(itersym, 1, (S.parameters[n])::Int))
#     end
#     elementbytes = sizeof(T)
#     LoopVectorization.add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
#     LoopVectorization.add_simple_store!(ls, :dest, LoopVectorization.ArrayReference(:dest, reverse(loopsyms)), elementbytes)
#     resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
#     q = LoopVectorization.lower(ls)
#     push!(q.args, :dest′)
#     pushfirst!(q.args, Expr(:meta,:inline))
#     q
#     # ls
# end
@inline function Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{S}) where {S <: AbstractStrideStyle}
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    Base.Broadcast.materialize!(similar(bc, ElType), bc)
end

LoopVectorization.vmaterialize(bc::Base.Broadcast.Broadcasted{<:AbstractStrideStyle}) = Base.Broadcast.materialize(bc)
LoopVectorization.vmaterialize!(dest, bc::Base.Broadcast.Broadcasted{<:AbstractStrideStyle}) = Base.Broadcast.materialize!(dest, bc)

LoopVectorization.vmaterialize(bc::StrideArrayProduct) = Base.Broadcast.materialize(bc)
LoopVectorization.vmaterialize!(dest, bc::StrideArrayProduct) = Base.Broadcast.materialize!(dest, bc)

Base.:(+)(A::AbstractStrideArray, B::AbstractStrideArray) = A .+ B
Base.:(-)(A::AbstractStrideArray, B::AbstractStrideArray) = A .- B

Base.unaliascopy(A::AbstractStrideArray) = A

