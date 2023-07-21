
using LoopVectorization: forbroadcast

_extract(::Type{StaticInt{N}}) where {N} = N::Int
_extract(_) = nothing

abstract type AbstractStrideStyle{S,N} <: Base.Broadcast.AbstractArrayStyle{N} end
struct LinearStyle{S,N,R} <: AbstractStrideStyle{S,N} end
struct CartesianStyle{S,N} <: AbstractStrideStyle{S,N} end
@generated function Base.BroadcastStyle(
  ::Type{A}
) where {T<:VectorizationBase.NativeTypes,N,R,S,X,A<:AbstractStrideArray{T,N,R,S,X}}
  t = Expr(:curly, :Tuple)
  for x ∈ X.parameters
    x === nothing || return CartesianStyle{S,N}()
  end
  for s ∈ Static.known(S)
    s === nothing ? push!(t.args, Int) : push!(t.args, s)
  end
  :(LinearStyle{$t,$N,$R}())
end

const StrideArrayProduct = Union{
  LoopVectorization.Product{<:AbstractStrideArray},
  LoopVectorization.Product{<:Any,<:AbstractStrideArray},
  LoopVectorization.Product{<:AbstractStrideArray,<:AbstractStrideArray}
}

@generated function Base.BroadcastStyle(
  ::Type{P}
) where {
  SA,
  A<:AbstractStrideArray{<:VectorizationBase.NativeTypes,<:Any,<:Any,SA},
  SB,
  B<:AbstractStrideArray{<:VectorizationBase.NativeTypes,<:Any,<:Any,SB},
  P<:LoopVectorization.Product{A,B}
}
  t = Expr(:curly, :Tuple)
  M = _extract(SA.parameters[1])
  D = length(SB.parameters)
  D ∈ (1, 2) || throw(
    ArgumentError(
      "In A*B, B should be a vector or matrix, but ndims(B) == $D."
    )
  )
  M === nothing ? push!(t.args, :Int) : push!(t.args, M)
  D == 1 && return :(CartesianStyle{$t,1}())
  N = _extract(SB.parameters[2])
  N === nothing ? push!(t.args, :Int) : push!(t.args, N)
  :(CartesianStyle{$t,2}())
end

@generated Base.BroadcastStyle(
  a::CartesianStyle{S,N1},
  ::Base.Broadcast.DefaultArrayStyle{N2}
) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : :a
@generated Base.BroadcastStyle(
  ::LinearStyle{S,N1},
  ::Base.Broadcast.DefaultArrayStyle{N2}
) where {S,N1,N2} = N2 > N1 ? Base.Broadcast.Unknown() : CartesianStyle{S,N1}()
Base.BroadcastStyle(
  a::CartesianStyle{S,N},
  ::AbstractStrideStyle{S,N}
) where {S,N} = a
Base.BroadcastStyle(
  ::AbstractStrideStyle{S,N},
  a::CartesianStyle{S,N}
) where {S,N} = a
Base.BroadcastStyle(a::CartesianStyle{S,N}, ::CartesianStyle{S,N}) where {S,N} =
  a # resolve ambiguities
Base.BroadcastStyle(a::LinearStyle{S,N,R}, ::LinearStyle{S,N,R}) where {S,N,R} =
  a # ranks match
Base.BroadcastStyle(::LinearStyle{S,N}, ::LinearStyle{S,N}) where {S,N} =
  CartesianStyle{S,N}() # ranks don't match
@generated function Base.BroadcastStyle(
  ::AbstractStrideStyle{S1,N1},
  ::AbstractStrideStyle{S2,N2}
) where {S1,S2,N1,N2}
  N2 > N1 && return :(Base.Broadcast.Unknown())
  S = Expr(:curly, :Tuple)
  for n ∈ 1:N2#min(N1,N2)
    _s1 = S1.parameters[n]
    _s2 = S2.parameters[n]
    s1 = (_s1 isa Int ? _s1 : -1)::Int
    s2 = (_s2 isa Int ? _s2 : -1)::Int
    if s1 == s2
      push!(S.args, s1)
    elseif s2 == 1
      push!(S.args, s1)
    elseif s1 == 1
      push!(S.args, s2)
    elseif s2 == -1
      push!(S.args, s1)
    elseif s1 == -1
      push!(S.args, s2)
    else
      throw("Mismatched sizes: $S1, $S2.")
    end
  end
  for n ∈ N2+1:N1
    push!(S.args, S1.parameters[n])
  end
  Expr(:call, Expr(:curly, :CartesianStyle, S, max(N1, N2)))
end

@generated function to_tuple(::Type{S}, s) where {N,S<:Tuple{Vararg{Any,N}}}
  t = Expr(:tuple)
  Sp = S.parameters
  for i = 1:length(Sp)
    l = _extract(Sp[i])
    if l === nothing
      push!(t.args, Expr(:ref, :s, i))
    else
      push!(t.args, StaticInt(l))
    end
  end
  t
end
function Base.similar(
  bc::Base.Broadcast.Broadcasted{FS},
  ::Type{T}
) where {S,T,N,FS<:AbstractStrideStyle{S,N}}
  StrideArray{T}(undef, to_tuple(S, static_size(bc)))
end
@generated function to_tuple(::Type{S}) where {N,S<:Tuple{Vararg{StaticInt,N}}}
  map(StaticInt, known(S))
end
function Base.similar(
  bc::Base.Broadcast.Broadcasted{FS},
  ::Type{T}
) where {S<:Tuple{Vararg{StaticInt}},T,N,FS<:AbstractStrideStyle{S,N}}
  StrideArray{T}(undef, to_tuple(S))
end

@inline _vecbc(x) = x
@inline _vecbc(x::AbstractArray) = vec(x)
@inline function _vecbc(bc::Base.Broadcast.Broadcasted)
  Base.Broadcast.Broadcasted(bc.f, map(_vecbc, bc.args))
end

@generated function _linear_matches(::Val{LS}, ::Val{S}) where {LS,S}
  (LS.parameters...,) === known(S)
end
# function Base.Broadcast.materialize!(
@inline function _materialize!(
  dest::AbstractStrideArray{<:Any,N,R,S},
  bc::BC,
  ::Val{UNROLL}
) where {
  S,
  N,
  R,
  LS,
  FS<:LinearStyle{LS,N,R},
  BC<:Base.Broadcast.Broadcasted{FS},
  UNROLL
}
  if _linear_matches(Val{LS}(), Val{S}())
    LoopVectorization.vmaterialize!(
      vec(dest),
      _vecbc(bc),
      Val{:StrideArrays}(),
      Val{UNROLL}()
    )
  else
    LoopVectorization.vmaterialize!(
      dest,
      bc,
      Val{:StrideArrays}(),
      Val{UNROLL}()
    )
  end
  return dest
end
@inline function _materialize!(
  # function _materialize!(
  dest::AbstractStrideArray{<:Any,N,R,S},
  bc::BC,
  ::Val{UNROLL}
) where {S,N,R,BC<:Union{Base.Broadcast.Broadcasted,StrideArrayProduct},UNROLL}
  LoopVectorization.vmaterialize!(dest, bc, Val{:StrideArrays}(), Val{UNROLL}())
end

@inline function Base.Broadcast.materialize!(
  dest::AbstractStrideArray,
  bc::BC
) where {BC<:Union{Base.Broadcast.Broadcasted,StrideArrayProduct}}
  _materialize!(
    dest,
    bc,
    LoopVectorization.avx_config_val(
      Val((
        true,
        zero(Int8),
        zero(Int8),
        zero(Int8),
        true,
        one(UInt),
        0,
        false
      )),
      pick_vector_width(eltype(dest))
    )
  )
end

@inline function Base.Broadcast.materialize!(
  dest::AbstractStrideArray,
  bc::Base.Broadcast.Broadcasted{
    Base.Broadcast.DefaultArrayStyle{0},
    Nothing,
    typeof(identity),
    Tuple{T}
  }
) where {T}
  fill!(dest, first(bc.args))
end

@inline function Base.Broadcast.materialize(
  bc::Base.Broadcast.Broadcasted{S}
) where {S<:AbstractStrideStyle}
  ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
  Base.Broadcast.materialize!(similar(bc, ElType), bc)
end

@inline LoopVectorization.vmaterialize(
  bc::Base.Broadcast.Broadcasted{<:AbstractStrideStyle}
) = Base.Broadcast.materialize(bc)
@inline LoopVectorization.vmaterialize!(
  dest,
  bc::Base.Broadcast.Broadcasted{<:AbstractStrideStyle}
) = Base.Broadcast.materialize!(dest, bc)

@inline LoopVectorization.vmaterialize(bc::StrideArrayProduct) =
  Base.Broadcast.materialize(bc)
@inline LoopVectorization.vmaterialize!(dest, bc::StrideArrayProduct) =
  Base.Broadcast.materialize!(dest, bc)

@inline Base.:(+)(A::AbstractStrideArray, B::AbstractStrideArray) = A .+ B
@inline Base.:(-)(A::AbstractStrideArray, B::AbstractStrideArray) = A .- B

@inline Base.unaliascopy(A::AbstractStrideArray) = A
