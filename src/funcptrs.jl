@inline function _atomic_load(p::Ptr{UInt}, offset, ::Type{T}) where {T}
    p2 = _atomic_load(p + sizeof(UInt)*(offset += 1))
    offset, reinterpret(T, p2)
end
@inline function _atomic_load(p::Ptr{UInt}, offset, ::Type{StaticInt{N}}) where {N}
    offset, StaticInt{N}()
end
@inline function _atomic_load(p::Ptr{UInt}, offset, ::Type{StridedPointer{T,2,C,B,R,Tuple{X1,X2},Tuple{Zero,Zero}}}) where {T,C,B,R,X1,X2}
    offset, ptr = _atomic_load(p, offset, Ptr{T})
    offset, x1 = _atomic_load(p, offset, X1)
    offset, x2 = _atomic_load(p, offset, X2)
    offset, StridedPointer{T,2,C,B,R}(ptr, (x1,x2), (Zero(),Zero()))
end

struct LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd} <: Function end
function (::LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd})(p::Ptr{UInt}) where {P,TC,TA,TB,Α,Β,Md,Kd,Nd}
    offset, C = _atomic_load(p, 1, TC)
    offset, A = _atomic_load(p, offset, TA)
    offset, B = _atomic_load(p, offset, TB)
    offset, α = _atomic_load(p, offset, Α)
    offset, β = _atomic_load(p, offset, Β)
    offset, M = _atomic_load(p, offset, Md)
    offset, K = _atomic_load(p, offset, Kd)
    offset, N = _atomic_load(p, offset, Nd)
    _call_loopmul!(C, A, B, α, β, M, K, N, Val{P}())
    nothing
end
@inline _call_loopmul!(C, A, B, α, β, M, K, N, ::Val{false}) = loopmul!(C, A, B, α, β, M, K, N)
@inline function _call_loopmul!(C::StridedPointer{T}, A, B, α, β, M, K, N, ::Val{true}) where {T}
    if M*K < first_effective_cache(T) * R₂Default
        packaloopmul!(C, A, B, α, β, M, K, N)
        return
    else
        jmulpackAonly!(C, A, B, α, β, M, K, N, StaticFloat{W₁Default}(), StaticFloat{W₂Default}(), StaticFloat{R₁Default}(), StaticFloat{R₂Default}())
        return
    end
end
call_loopmul!(C, A, B, α, β, M, K, N, ::Val{P}) where {P} = _call_loopmul!(C, A, B, α, β, M, K, N, Val{P}())

struct SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂} <: Function end
function (::SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂})(p::Ptr{UInt}) where {TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}
    offset, C = _atomic_load(p, 1, TC)
    offset, A = _atomic_load(p, offset, TA)
    offset, B = _atomic_load(p, offset, TB)
    offset, α = _atomic_load(p, offset, Α)
    offset, β = _atomic_load(p, offset, Β)
    offset, M = _atomic_load(p, offset, Md)
    offset, K = _atomic_load(p, offset, Kd)
    offset, N = _atomic_load(p, offset, Nd)
    offset, atomicp = _atomic_load(p, offset, AP)
    offset, bcachep = _atomic_load(p, offset, BCP)
    offset, id = _atomic_load(p, offset, ID)
    offset, total_ids = _atomic_load(p, offset, TT)
    sync_mul!(C, A, B, α, β, M, K, N, atomicp, bcachep, id, total_ids, StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}())
    nothing
end


@generated function cfuncpointer(::T) where {T}
    precompile(T(), (Ptr{UInt},))
    quote
        @cfunction($(T()), Cvoid, (Ptr{UInt},))
    end
end


@inline function _atomic_store!(p::Ptr{UInt}, x, offset)
    _atomic_store!(p + sizeof(UInt)*(offset += 1), reinterpret(UInt, x))
    offset
end
@inline function _atomic_store!(p::Ptr{UInt}, sp::StridedPointer{T,2}, offset) where {T}
    offset = _atomic_store!(p, sp.p, offset)
    x1, x2 = sp.strd
    offset = _atomic_store!(p, x1, offset)
    offset = _atomic_store!(p, x2, offset)
end
@inline _atomic_store!(p::Ptr{UInt}, ::StaticInt, offset) = offset

function setup_matmul!(p::Ptr{UInt}, C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, ::Val{P}) where {P,TC,TA,TB,Α,Β,Md,Kd,Nd}
    offset = _atomic_store!(p, cfuncpointer(LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd}()), 0)
    offset = _atomic_store!(p, C, offset)
    offset = _atomic_store!(p, A, offset)
    offset = _atomic_store!(p, B, offset)
    offset = _atomic_store!(p, α, offset)
    offset = _atomic_store!(p, β, offset)
    offset = _atomic_store!(p, M, offset)
    offset = _atomic_store!(p, K, offset)
    offset = _atomic_store!(p, N, offset)
    nothing
end

function setup_syncmul!(
    p::Ptr{UInt}, C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd,
    ap::AP,bcp::BCP,id::ID,tt::TT,::StaticFloat{W₁},::StaticFloat{W₂},::StaticFloat{R₁},::StaticFloat{R₂}
) where {TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}
    offset = _atomic_store!(p, cfuncpointer(SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}()), 0)
    offset = _atomic_store!(p, C, offset)
    offset = _atomic_store!(p, A, offset)
    offset = _atomic_store!(p, B, offset)
    offset = _atomic_store!(p, α, offset)
    offset = _atomic_store!(p, β, offset)
    offset = _atomic_store!(p, M, offset)
    offset = _atomic_store!(p, K, offset)
    offset = _atomic_store!(p, N, offset)
    offset = _atomic_store!(p, ap,  offset)
    offset = _atomic_store!(p, bcp, offset)
    offset = _atomic_store!(p, id,  offset)
    offset = _atomic_store!(p, tt,  offset)
    nothing
end

function _matmul!(p::Ptr{UInt})
    _, fptr = _atomic_load(p, 0, Ptr{Cvoid})
    ccall(fptr, Cvoid, (Ptr{UInt},), p)
end

