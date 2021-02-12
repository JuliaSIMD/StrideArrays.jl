module StrideArrays

# Write your package code here.

using VectorizationBase, ArrayInterface,
    SLEEFPirates, VectorizedRNG,
    LoopVectorization, LinearAlgebra,
    Random#, StackPointers#,
    # SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase: align, gep, AbstractStridedPointer, AbstractSIMDVector, vnoaliasstore!, staticm1,
    static_sizeof, lazymul, vmul_fast, StridedPointer, gesp, zero_offsets, pause, zstridedpointer,
    val_dense_dims, val_stride_rank, preserve_buffer
using LoopVectorization: maybestaticsize, CloseOpen
using ArrayInterface: StaticInt, Zero, One, StaticBool, True, False,
    OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes,
    dense_dims, stride_rank
using StrideArraysCore: AbstractStrideArray, AbstractStrideMatrix, AbstractStrideVector,
    AbstractPtrStrideArray, PtrArray, static_expr, rank_to_sortperm

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
# include("stridedpointers.jl")
include("initialization.jl")
include("rand.jl")
include("blas.jl")
include("broadcast.jl")
include("miscellaneous.jl")


# include("precompile.jl")
# _precompile_()


end
