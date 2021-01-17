
const BCACHE_COUNT = something(VectorizationBase.CACHE_COUNT[3], 1);
const BSIZE = Int(something(cache_size(Float64, Val(3)), 393216));

const BCACHE_LOCK = Atomic{UInt}(zero(UInt))

struct BCache{T<:Union{UInt,Nothing}}
    p::Ptr{Float64}
    i::T
end
BCache(i::Integer) = BCache(pointer(BCACHE)+8cld_fast(BSIZE*i,Threads.nthreads()), i % UInt)
BCache(::Nothing) = BCache(pointer(BCACHE), nothing)

@inline Base.pointer(b::BCache) = b.p
@inline Base.unsafe_convert(::Type{Ptr{T}}, b::BCache) where {T} = Base.unsafe_convert(Ptr{T}, b.p)


function _use_bcache()
    while atomic_cas!(BCACHE_LOCK, zero(UInt), typemax(UInt)) != zero(UInt)
        pause()
    end
    return BCache(nothing)
end
@inline _free_bcache!(b::BCache{Nothing}) = reseet_bcache_lock!()

_use_bcache(::Nothing) = _use_bcache()
function _use_bcache(i)
    f = one(UInt) << i
    while (atomic_or!(BCACHE_LOCK, f) & f) != zero(UInt)
        pause()
    end
    BCache(i)
end
_free_bcache!(b::BCache{UInt}) = (atomic_xor!(BCACHE_LOCK, one(UInt) << b.i); nothing)

"""
  reset_bcache_lock!()

Currently not using try/finally in matmul routine, despite locking.
So if it errors for some reason, you may need to manually call `reset_bcache_lock!()`.
"""
@inline reseet_bcache_lock!() = (BCACHE_LOCK[] = zero(UInt); nothing)




