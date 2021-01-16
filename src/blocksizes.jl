const FIRST__CACHE = 1 + (VectorizationBase.CACHE_SIZE[3] !== nothing)
const SECOND_CACHE = 2 + (VectorizationBase.CACHE_SIZE[3] !== nothing)
const FIRST__CACHE_SIZE = VectorizationBase.CACHE_SIZE[FIRST__CACHE] === nothing ? 262144 :
    (((FIRST__CACHE == 2) & CACHE_INCLUSIVITY[2]) ? (VectorizationBase.CACHE_SIZE[2] - VectorizationBase.CACHE_SIZE[1]) :
    VectorizationBase.CACHE_SIZE[FIRST__CACHE])
const SECOND_CACHE_SIZE = (VectorizationBase.CACHE_SIZE[SECOND_CACHE] === nothing ? 3145728 :
    (CACHE_INCLUSIVITY[SECOND_CACHE] ? (VectorizationBase.CACHE_SIZE[SECOND_CACHE] - VectorizationBase.CACHE_SIZE[FIRST__CACHE]) :
    VectorizationBase.CACHE_SIZE[SECOND_CACHE])) * something(VectorizationBase.CACHE_COUNT[SECOND_CACHE], 1)



first_effective_cache(::Type{T}) where {T} = StaticInt{FIRST__CACHE_SIZE}() ÷ static_sizeof(T)
second_effective_cache(::Type{T}) where {T} = StaticInt{SECOND_CACHE_SIZE}() ÷ static_sizeof(T)

const W₁Default = 0.006163438737441861
const W₂Default = 0.7780817655617109
const R₁Default = 0.5264474314966798
const R₂Default = 0.7537004634660829

function matmul_params(::Type{T}, _α, _β, R₁, R₂) where {T}
    W = VectorizationBase.pick_vector_width_val(T)
    α = _α * W
    β = _β * W
    L₁ₑ = first_effective_cache(T) * R₁
    L₂ₑ = second_effective_cache(T) * R₂
    MᵣW = StaticInt{mᵣ}() * W
    
    Mc = floortostaticint(√(L₁ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₂ₑ) / MᵣW) * MᵣW
    Kc = roundtostaticint(√(L₁ₑ)*√(L₂ₑ)/√(L₁ₑ*β + L₂ₑ*α))
    Nc = floortostaticint(√(L₂ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₁ₑ) / StaticInt{nᵣ}()) * StaticInt{nᵣ}()

    Mc, Kc, Nc
end
function matmul_params(::Type{T}) where {T}
    matmul_params(T, StaticFloat{W₁Default}(), StaticFloat{W₂Default}(), StaticFloat{R₁Default}(), StaticFloat{R₂Default}())
end

"""
    split_m(M, Miters_base, W)

Splits `M` into at most `Miters_base` iterations.
For example, if we wish to divide `517` iterations into roughly 7 blocks using multiples of `8`:

```julia
julia> split_m(517, 7, 8)
(72, 2, 69, 7)
```

This suggests we have base block sizes of size `72`, with two iterations requiring an extra remainder of `8 ( = W)`,
and a final block of `69` to handle the remainder. It also tells us that there are `7` total iterations, as requested.
```julia
julia> 80*2 + 72*(7-2-1) + 69
517
```
This is meant to specify roughly the requested amount of blocks, and return relatively even sizes.

This method is used fairly generally.
"""
@inline function split_m(M, _Mblocks, W)
    Miters = cld_fast(M, W)
    Mblocks = min(_Mblocks, Miters)
    Miter_per_block, Mrem = divrem_fast(Miters, Mblocks)
    Mbsize = Miter_per_block * W
    Mremfinal = M - Mbsize*(Mblocks-1) - Mrem * W
    Mbsize, Mrem, Mremfinal, Mblocks
end
# @inline function split_m(M, _Mblocks, W, Mᵣ)
#     MᵣW = Mᵣ * W
#     mi = cld_fast(M, _Mblocks * MᵣW)
#     Miters = cld_fast(M, W)
#     _Miter = cld_fast(M, MᵣW*mi)
#     split_m(M, _Miter, W)
#     Mbsize = divrem_fast(Miters, _Miter)
# end

"""
  solve_block_sizes(::Type{T}, M, K, N, α, β, R₂, R₃)

This function returns iteration/blocking descriptions `Mc`, `Kc`, and `Nc` for use when packing both `A` and `B`.

It tries to roughly minimize the cost
```julia
MKN/(Kc*W) + α * MKN/Mc + β * MKN/Nc
```
subject to constraints
```julia
Mc - M ≤ 0
Kc - K ≤ 0
Nc - N ≤ 0
Mc*Kc - L₁ₑ ≤ 0
Kc*Nc - L₂ₑ ≤ 0
```
That is, our constraints say that our block sizes shouldn't be bigger than the actual dimensions, and also that
our packed `A` (`Mc × Kc`) should fit into the first packing cache (generally, actually the `L₂`, and our packed
`B` (`Kc × Nc`) should fit into the second packing cache (generally the `L₃`).

Our cost model consists of three components:
1. Cost of moving data in and out of registers. This is done `(M/Mᵣ * K/Kc * N/Nᵣ)` times and the cost per is `(Mᵣ/W * Nᵣ)`.
2. Cost of moving strips from `B` pack from the low cache levels to the highest cache levels when multiplying `Aₚ * Bₚ`.
   This is done `(M / Mc * K / Kc * N / Nc)` times, and the cost per is proportional to `(Kc * Nᵣ)`.
   `α` is the proportionality-constant parameter.
3. Cost of packing `A`. This is done `(M / Mc * K / Kc * N / Nc)` times, and the cost per is proportional to
   `(Mc * Kc)`. `β` is the proportionality-constant parameter.

As `W` is a constant, we multiply the cost by `W` and absorb it into `α` and `β`. We drop it from the description
from  here on out.

In the full problem, we would have Lagrangian, with μ < 0:
f((Mc,Kc,Nc),(μ₁,μ₂,μ₃,μ₄,μ₅))
MKN/Kc + α * MKN/Mc + β * MKN/Nc - μ₁(Mc - M) - μ₂(Kc - K) - μ₃(Nc - N) - μ₄(Mc*Kc - L2) - μ₅(Kc*Nc - L3)
```julia
0 = ∂L/∂Mc = - α * MKN / Mc² - μ₁ - μ₄*Kc
0 = ∂L/∂Kc = - MKN / Kc² - μ₂ - μ₄*Mc - μ₅*Nc
0 = ∂L/∂Nc = - β * MKN / Nc² - μ₃ - μ₅*Kc
0 = ∂L/∂μ₁ = M - Mc
0 = ∂L/∂μ₂ = K - Kc
0 = ∂L/∂μ₃ = N - Nc
0 = ∂L/∂μ₄ = L₁ₑ - Mc*Kc
0 = ∂L/∂μ₅ = L₂ₑ - Kc*Nc
```
The first 3 constraints complicate things, because they're trivially solved by setting `M = Mc`, `K = Kc`, and `N = Nc`.
But this will violate the last two constraints in general; normally we will be on the interior of the inequalities,
meaning we'd be dropping those constraints. Doing so, this leaves us with:

First, lets just solve the cost w/o constraints 1-3
```julia
0 = ∂L/∂Mc = - α * MKN / Mc² - μ₄*Kc
0 = ∂L/∂Kc = - MKN / Kc² - μ₄*Mc - μ₅*Nc
0 = ∂L/∂Nc = - β * MKN / Nc² - μ₅*Kc
0 = ∂L/∂μ₄ = L₁ₑ - Mc*Kc
0 = ∂L/∂μ₅ = L₂ₑ - Kc*Nc
```
Solving:
```julia
Mc = √(L₁ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₂ₑ)
Kc = √(L₁ₑ)*√(L₂ₑ)/√(L₁ₑ*β + L₂ₑ*α)
Nc = √(L₂ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₁ₑ)
μ₄ = -K*√(L₂ₑ)*M*N*α/(L₁ₑ^(3/2)*√(L₁ₑ*β + L₂ₑ*α))
μ₅ = -K*√(L₁ₑ)*M*N*β/(L₂ₑ^(3/2)*√(L₁ₑ*β + L₂ₑ*α))
```
These solutions are indepedent of matrix size.
The approach we'll take here is solving for `Nc`, `Kc`, and then finally `Mc` one after the other, incorporating sizes.

Starting with `N`, we check how many iterations would be implied by `Nc`, and then choose the smallest value that would
yield that number of iterations. This also ensures that `Nc ≤ N`.
```julia
Niter = cld(N, √(L₂ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₁ₑ))
Nblock, Nrem = divrem(N, Niter)
Nblock_Nrem = Nblock + (Nrem > 0)
```
We have `Nrem` iterations of size `Nblock_Nrem`, and `Niter - Nrem` iterations of size `Nblock`.

We can now make `Nc = Nblock_Nrem` a constant, and solve the remaining three equations again:
```julia
0 = ∂L/∂Mc = - α * MKN / Mc² - μ₄*Kc
0 = ∂L/∂Kc = - MKN / Kc² - μ₄*Mc - μ₅*Ncm
0 = ∂L/∂μ₄ = L₂ₑ - Mc*Kc
```
yielding
```julia
Mc = √(L₁ₑ)*√(α)
Kc = √(L₁ₑ)/√(α)
μ₄ = -K*M*N*√(α)/L₁ₑ^(3/2)
```
We proceed in the same fashion as for `Nc`, being sure to reapply the `Kc * Nc ≤ L₂ₑ` constraint:
```julia
Kiter = cld(K, min(√(L₁ₑ)/√(α), L₂ₑ/Nc))
Kblock, Krem = divrem(K, Ki)
Kblock_Krem = Kblock + (Krem > 0)
```
This leaves `Mc` partitioning, for which, for which we use the constraint `Mc * Kc ≤ L₁ₑ` to set
the initial number of proposed iterations as `cld(M, L₁ₑ / Kcm)` for calling `split_m`.
```julia
Mbsize, Mrem, Mremfinal, Mblocks = split_m(M, cld(M, L₁ₑ / Kcm), StaticInt{W}())
```

Note that for synchronization on `B`, all threads must have the same values for `Kc` and `Nc`.
`K` and `N` will be equal between threads, but `M` may differ. By calculating `Kc` and `Nc`
independently of `M`, this algorithm guarantees all threads are on the same page.
"""
@inline function solve_block_sizes(::Type{T}, M, K, N, _α, _β, R₂, R₃, Wfactor) where {T}
    W = VectorizationBase.pick_vector_width_val(T)
    α = _α * W
    β = _β * W
    L₁ₑ =  first_effective_cache(T) * R₂
    L₂ₑ = second_effective_cache(T) * R₃

    # Nc_init = round(Int, √(L₂ₑ)*√(α * L₂ₑ + β * L₁ₑ)/√(L₁ₑ))
    Nc_init⁻¹ = √(L₁ₑ) / (√(L₂ₑ)*√(α * L₂ₑ + β * L₁ₑ))
    
    Niter = cldapproxi(N, Nc_init⁻¹) # approximate `ceil`
    Nblock, Nrem = divrem_fast(N, Niter)
    Nblock_Nrem = Nblock + One()#(Nrem > 0)

    ((Mblock, Mblock_Mrem, Mremfinal, Mrem, Miter), (Kblock, Kblock_Krem, Krem, Kiter)) = solve_McKc(T, M, K, Nblock_Nrem, _α, _β, R₂, R₃, Wfactor)
    
    (Mblock, Mblock_Mrem, Mremfinal, Mrem, Miter), (Kblock, Kblock_Krem, Krem, Kiter), promote(Nblock, Nblock_Nrem, Nrem, Niter)
end
# Takes Nc, calcs Mc and Kc
@inline function solve_McKc(::Type{T}, M, K, Nc, _α, _β, R₂, R₃, Wfactor) where {T}
    W = VectorizationBase.pick_vector_width_val(T)
    α = _α * W
    β = _β * W
    L₁ₑ =  first_effective_cache(T) * R₂
    L₂ₑ = second_effective_cache(T) * R₃

    Kc_init⁻¹ = Base.FastMath.max_fast(√(α/L₁ₑ), Nc*inv(L₂ₑ))
    Kiter = cldapproxi(K, Kc_init⁻¹) # approximate `ceil`
    Kblock, Krem = divrem_fast(K, Kiter)
    Kblock_Krem = Kblock + One()

    Miter_init = cldapproxi(M * inv(L₁ₑ), Kblock_Krem) # Miter = M * Kc / L₁ₑ
    Mbsize, Mrem, Mremfinal, Miter = split_m(M, Miter_init, W * Wfactor)
    Mblock_Mrem = Mbsize + W * Wfactor
    
    promote(Mbsize, Mblock_Mrem, Mremfinal, Mrem, Miter), promote(Kblock, Kblock_Krem, Krem, Kiter)
end

@inline cldapproxi(n, d⁻¹) = Base.fptosi(Int, Base.FastMath.add_fast(Base.FastMath.mul_fast(n, d⁻¹), 0.9999999999999432)) # approximate `ceil`

# @inline function gcd_fast(a::T, b::T) where {T<:Base.BitInteger}
#     za = trailing_zeros(a)
#     zb = trailing_zeros(b)
#     k = min(za, zb)
#     u = unsigned(abs(a >> za))
#     v = unsigned(abs(b >> zb))
#     while u != v
#         if u > v
#             u, v = v, u
#         end
#         v -= u
#         v >>= trailing_zeros(v)
#     end
#     r = u << k
#     r % T
# end

"""
  find_first_acceptable(M, W)

Finds first combination of `Miter` and `Niter` that doesn't make `M` too small while producing `Miter * Niter = NUM_CORES`.
This would be awkard if there are computers with prime numbers of cores. I should probably consider that possibility at some point.
"""
@inline function find_first_acceptable(M, W)
    Mᵣ = StaticInt{mᵣ}() * W
    for (miter,niter) ∈ CORE_FACTORS
        if miter * ((MᵣW_mul_factor - One()) * Mᵣ) ≤ M + (W + W)
            return miter, niter
        end
    end
    last(CORE_FACTORS)
end
"""
  divide_blocks(M, Ntotal, _nspawn, W)

Splits both `M` and `N` into blocks when trying to spawn a large number of threads relative to the size of the matrices.
"""
@inline function divide_blocks(M, Ntotal, _nspawn, W)
    _nspawn == NUM_CORES && return find_first_acceptable(M, W)
    
    Miter = clamp(div_fast(M, W*StaticInt{mᵣ}() * MᵣW_mul_factor), 1, _nspawn)
    nspawn = div_fast(_nspawn, Miter)
    Niter = if (nspawn ≤ 1) & (Miter < _nspawn)
        # rebalance Miter
        Miter = cld_fast(_nspawn, cld_fast(_nspawn, Miter))
        nspawn = div_fast(_nspawn, Miter)
    end
    Miter, cld_fast(Ntotal, max(2, cld_fast(Ntotal, nspawn)))
end

