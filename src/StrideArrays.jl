module StrideArrays

# Write your package code here.

using VectorizationBase,
    ArrayInterface, SLEEFPirates, VectorizedRNG, LoopVectorization, LinearAlgebra, Random#, StackPointers#,
# SpecialFunctions # Perhaps there is a better way to support erf?

using VectorizationBase:
    align,
    AbstractStridedPointer,
    AbstractSIMDVector,
    StridedPointer,
    gesp,
    pause,
    zstridedpointer,
    val_dense_dims,
    preserve_buffer
using LoopVectorization: CloseOpen
using Static: StaticInt, Zero, One, StaticBool, True, False, known
using ArrayInterface:
    size,
    strides,
    offsets,
    indices,
    static_length,
    axes,
    dense_dims,
    stride_rank,
    StrideIndex
using StrideArraysCore:
    AbstractStrideArray,
    AbstractStrideMatrix,
    AbstractStrideVector,
    AbstractPtrStrideArray,
    PtrArray,
    rank_to_sortperm,
    StrideArray,
    StrideVector,
    StrideMatrix,
    similar_layout,
    @gc_preserve

using Octavian
using Octavian: MemoryBuffer

export @StrideArray,
    @gc_preserve, # @Constant,
    AbstractStrideArray,
    AbstractStrideVector,
    AbstractStrideMatrix,
    StrideArray,
    StrideVector,
    StrideMatrix,
    PtrArray,# PtrVector, PtrMatrix,
    # ConstantArray, ConstantVector, ConstantMatrix, allocarray,
    matmul!,
    matmul_serial!,
    mul!,
    *ˡ,
    StaticInt,
    matmul,
    matmul_serial
# LazyMap, 

include("rand.jl")
include("blas.jl")
include("broadcast.jl")
include("miscellaneous.jl")


# include("precompile.jl")
# _precompile_()


end
