module StrideArrays

# Write your package code here.

using VectorizationBase, ArrayInterface,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random, Base.Threads#, StackPointers#,
    # SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: align, gep, AbstractStridedPointer, AbstractSIMDVector, vnoaliasstore!, staticm1,
    static_sizeof, lazymul, vmul_fast, StridedPointer, gesp, zero_offsets, pause,
    CACHE_COUNT, NUM_CORES, CACHE_INCLUSIVITY, zstridedpointer
using LoopVectorization: maybestaticsize, mᵣ, nᵣ, preserve_buffer, CloseOpen
using ArrayInterface: StaticInt, Zero, One, OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes,
    dense_dims, DenseDims, stride_rank, StrideRank
# using Threads: @spawn
# import ReverseDiffExpressionsBase:
    # RESERVED_INCREMENT_SEED_RESERVED!, ∂getindex,
    # alloc_adjoint, uninitialized, initialized, isinitialized
# import LoopVectorization: isdense

# using Parameters: @unpack

export @StrideArray, @gc_preserve, # @Constant,
    AbstractStrideArray, AbstractStrideVector, AbstractStrideMatrix,
    StrideArray, StrideVector, StrideMatrix,
    PtrArray,# PtrVector, PtrMatrix,
    # ConstantArray, ConstantVector, ConstantMatrix, allocarray,
    matmul!, matmul_serial!, mul!, *ˡ, StaticInt,
    matmul, matmul_serial
# LazyMap, 



include("type_declarations.jl")
include("staticfloats.jl")

include("funcptrs.jl")
include("threadpool.jl")

include("l3_cache_buffer.jl")
include("size_and_strides.jl")
include("adjoints.jl")
include("stridedpointers.jl")
include("indexing.jl")
include("initialization.jl")
include("views.jl")
include("rand.jl")
include("blocksizes.jl")
include("kernels.jl")
include("blas.jl")
include("broadcast.jl")
include("miscellaneous.jl")


# Commented, because I don't want this to be the only doc string.
# """
# To find a mode, define methods for `logdensity` and logdensity_and_gradient!` dispatching on obj, and evaluating at the position `q`.

# logdensity(obj, q, [::StackPointer])
# ∂logdensity!(∇, obj, q, [::StackPointer])

# These must return a value (eg, a logdensity). logdensity_and_gradient! should store the gradient in ∇.
# """
function logdensity end
function ∂logdensity! end

@generated function calc_factors(::Val{nc} = Val{NUM_CORES}()) where {nc}
    t = Expr(:tuple)
    for i ∈ nc:-1:1
        d, r = divrem(nc, i)
        iszero(r) && push!(t.args, (i, d))
    end
    t
end
const CORE_FACTORS = calc_factors()


const BCACHE = Float64[]
# """
# Length is one less than `Base.nthreads()`
# """
const MULTASKS = Task[]
const NTHREAD = Ref{Int}()
_nthreads() = NTHREAD[]

# function runfunc(t::Task, tid)
#     t.sticky = true
#     ccall(:jl_set_task_tid, Cvoid, (Any, Cint), t, tid)
#     push!(@inbounds(Base.Workqueues[tid+1]), t)
#     ccall(:jl_wakeup_thread, Cvoid, (Int16,), tid % Int16)
#     t
# end
# runfunc(func, tid) = runfunc(Task(func), tid)
# function runfunc!(ft, tid)
#     @inbounds MULTASKS[tid] = runfunc(ft, tid)
#     nothing
# end

function __init__()
    resize!(BCACHE, BSIZE * BCACHE_COUNT)
    NTHREAD[] = _nt = min(nthreads(), NUM_CORES)
    _nt -= 1
    iszero(_nt) || resize!(MULTASKS, _nt)
    if _nt < NUM_CORES - 1 && ("SUPPRESS_STRIDE_ARRAYS_WARNING" ∉ keys(ENV))
        msg = string(
            "Your system has $NUM_CORES physical cores, but `StrideArrays.jl` only has ",
            "$(_nt > 0 ? "$(_nt + 1) threads" : "1 thread") available.",
            "For the best performance, you should start Julia with at least $(NUM_CORES) threads.",
            "",
        )
        @warn msg
    end
    for tid ∈ 1:_nt
        m = MATMULLERS[tid]
        GC.@preserve m _atomic_min!(pointer(m), SPIN)
        t = Task(m)
        t.sticky = true
        ccall(:jl_set_task_tid, Cvoid, (Any, Cint), t, tid % Cint)
        MULTASKS[tid] = t
        wake_thread!(tid) # task should immediately sleep
        while !_atomic_cas_cmp!(pointer(m), WAIT, WAIT)
            pause()
        end
    end
end

# include("precompile.jl")
# _precompile_()


end
