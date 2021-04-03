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
using Static: StaticInt, Zero, One, StaticBool, True, False
using ArrayInterface: OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes,
    dense_dims, stride_rank
using StrideArraysCore: AbstractStrideArray, AbstractStrideMatrix, AbstractStrideVector,
    AbstractPtrStrideArray, PtrArray, static_expr, rank_to_sortperm,
    StrideArray, StrideVector, StrideMatrix, similar_layout,
    @gc_preserve

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

include("rand.jl")
include("blas.jl")
include("broadcast.jl")
include("miscellaneous.jl")


# include("precompile.jl")
# _precompile_()


end
