
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


for (ityp,jtyp) ∈ [("i32", UInt32), ("i64", UInt64), ("i128", UInt128)]
    @eval begin
        @inline function _atomic_load(ptr::Ptr{$jtyp})
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = load atomic volatile $(ityp), $(ityp)* %p acquire, align $(Base.gc_alignment(jtyp))
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}}, ptr)
        end
        @inline function _atomic_store!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              store atomic volatile $(ityp) %1, $(ityp)* %p release, align $(Base.gc_alignment(jtyp))
              ret void
            """), Cvoid, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
        @inline function _atomic_cas_cmp!(ptr::Ptr{$jtyp}, cmp::$jtyp, newval::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %c = cmpxchg volatile $(ityp)* %p, $(ityp) %1, $(ityp) %2 acq_rel acquire
              %bit = extractvalue { $ityp, i1 } %c, 1
              %bool = zext i1 %bit to i8
              ret i8 %bool
            """), Bool, Tuple{Ptr{$jtyp}, $jtyp, $jtyp}, ptr, cmp, newval)
        end
        @inline function _atomic_add!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile add $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
        @inline function _atomic_nand!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile nand $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
        @inline function _atomic_and!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile and $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
        @inline function _atomic_or!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile or $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
        @inline function _atomic_xor!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile xor $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
        @inline function _atomic_max!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile umax $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
        @inline function _atomic_min!(ptr::Ptr{$jtyp}, x::$jtyp)
            Base.llvmcall($("""
              %p = inttoptr $(ityp) %0 to $(ityp)*
              %v = atomicrmw volatile umin $(ityp)* %p, $(ityp) %1 acq_rel
              ret $(ityp) %v
            """), $jtyp, Tuple{Ptr{$jtyp}, $jtyp}, ptr, x)
        end
    end
end


