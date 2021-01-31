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

using ThreadingUtilities:
    _atomic_add!, _atomic_max!, _atomic_min!,
    _atomic_load, _atomic_store!, _atomic_cas_cmp!,
    SPIN, WAIT, TASK, LOCK, STUP, taskpointer,
    wake_thread!, __wait

using Octavian
using Octavian: MemoryBuffer

export @StrideArray, @gc_preserve, # @Constant,
    AbstractStrideArray, AbstractStrideVector, AbstractStrideMatrix,
    StrideArray, StrideVector, StrideMatrix,
    PtrArray,# PtrVector, PtrMatrix,
    # ConstantArray, ConstantVector, ConstantMatrix, allocarray,
    matmul!, matmul_serial!, mul!, *ˡ, StaticInt,
    matmul, matmul_serial
# LazyMap, 

include("type_declarations.jl")
include("size_and_strides.jl")
include("adjoints.jl")
include("stridedpointers.jl")
include("indexing.jl")
include("initialization.jl")
include("views.jl")
include("rand.jl")
include("blas.jl")
include("broadcast.jl")
include("miscellaneous.jl")


# include("precompile.jl")
# _precompile_()


end
