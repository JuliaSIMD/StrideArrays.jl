@inline ArrayInterface.size(A::StrideArray) = getfield(getfield(A, :ptr), :size)

@inline VectorizationBase.bytestrides(A::StrideArray) = getfield(getfield(getfield(A, :ptr), :ptr), :strd)
@inline ArrayInterface.strides(A::StrideArray) = strides(getfield(A, :ptr))
@inline ArrayInterface.offsets(A::StrideArray) = getfield(getfield(getfield(A, :ptr), :ptr), :offsets)

@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{One}) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::Base.OneTo) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::AbstractUnitRange) = Zero():(last(r)-first(r))

@inline zeroindex(r::CloseOpen{Zero}) = r
@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{Zero}) = r


