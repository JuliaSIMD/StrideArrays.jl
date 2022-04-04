# @inline function matmul_axes(C, A, B)
#     M = indices((C,A), (StaticInt{1}(),StaticInt{1}()))
#     K = indices((A,B), (StaticInt{2}(),StaticInt{1}()))
#     N = indices((C,B), (StaticInt{2}(),StaticInt{2}()))
#     M, K, N
# end

# @inline _select(::StaticInt{M}, ::StaticInt{M}) where {M} = StaticInt{M}()
# @noinline _select(::StaticInt{M}, ::StaticInt{N}) where {M,N} = throw("$M ≠ $N")
# @inline _select(::StaticInt{M}, _) where {M} = StaticInt{M}()
# @inline _select(_, ::StaticInt{M}) where {M} = StaticInt{M}()
# @inline _select(x, _) = x
# @inline function matmul_sizes(C,A,B)
#     MC, NC = size(C)
#     MA, KA = size(A)
#     KB, NB = size(B)
#     @assert ((MC == MA) & (KA == KB) & (NC == NB)) "Size mismatch."
#     (_select(MA, MC), _select(KA, KB), _select(NB, NC))
# end

# function two_cols_per_vector_add_kn_block!(q::Expr, Kunroll, Nunroll, Koffset, Noffset, Ashuffle, Bshuffle, MindA, MindC, Amask)
#     for kk ∈ 0:Kunroll-1
#         kt = kk + Koffset
#         Aload = Expr(:call, :vload, :ptrA, Expr(:tuple, MindA, kt))
#         isnothing(Amask) || push!(Aload.args, Amask)
#         push!(q.args, Expr(:(=), Symbol(:A_,kk), Expr(:call, :shufflevector, Expr(:call, :extract_data, Aload), Ashuffle)))
#     end
#     for kk ∈ 0:Kunroll-1, nn ∈ 0:Nunroll-1
#         kt = kk + Koffset
#         nt1 = 2(nn + Noffset)
#         nt2 = nt1 + 1
#         Bsym = Symbol(:B_,kk,:_,nn)
#         Csym = Symbol(:C_, nn)
#         # push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt)), Bshuffle)))
#         shuffle = Expr(
#             :tuple,
#             Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt1))),
#             Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2)))
#         )
#         push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, shuffle, Bshuffle)))
#         if iszero(Koffset | kk)
#             push!(q.args, Expr(:(=), Csym, Expr(:call, :vmul_fast, Symbol(:A_, kk), Bsym)))
#         else
#             push!(q.args, Expr(:(=), Csym, Expr(:call, :vfmadd, Symbol(:A_, kk), Bsym, Csym)))
#         end
#     end
#     for nn ∈ 0:Nunroll-1
#         push!(q.args, Expr(:call, :vnoaliasstore!, :ptrC, Symbol(:C_, nn), Expr(:tuple, MindC, 2(nn + Noffset))))
#     end
# end

# function initmulquote()
#     Expr(
#         :block,
#         Expr(:meta, :inline),
#         Expr(:(=), :ptrC, Expr(:call, :stridedpointer, :C)),
#         Expr(:(=), :ptrA, Expr(:call, :stridedpointer, :A)),
#         Expr(:(=), :ptrB, Expr(:call, :stridedpointer, :B))
#     )
# end

# function two_cols_per_vector_quote(K, N, W, Wshift, Amask = nothing)
#     # Calculate 2 columns of C at a time
#     q = two_cols_per_vector_quote!(initmulquote(), K, N, W, Wshift, 0, Amask)
#     push!(q.args, :C)
#     q
# end

# function two_cols_per_vector_quote!(q, K, N, W, Wshift, Noffbase = 0, Amask = nothing)
#     Ndoublereps = N >> 1
#     Wh = W >>> 1
#     Whs = Wshift - 1

#     Kloop = LoopVectorization.Loop(:k, 1, K, Symbol(""), Symbol(""), true, true)
#     Nloop = LoopVectorization.Loop(:n, 1, Ndoublereps, Symbol(""), Symbol(""), true, true)
#     cost_vec     = Float64[ 64.0, 32.0, 8.0, 0.0 ] # Values chosen to hopefully produce desired behavior... TODO: do this better?
#     reg_pressure = Float64[ 1.0, 1.0, 1.0, 0.0, 0.0 ]
#     Kunroll, Nunroll, cost = LoopVectorization.solve_unroll(
#         :k, :n, cost_vec, reg_pressure, W, :m, Kloop, Nloop, 1, 1
#     )
#     @assert isfinite(cost)
    
#     Nrep, Nrem = divrem(Ndoublereps, Nunroll)
#     Krep, Krem = divrem(K, Kunroll)
#     Ashuftup = Expr(:tuple); append!(Ashuftup.args, 0:Wh-1); append!(Ashuftup.args, 0:Wh-1)
#     Bshuftup = Expr(:tuple); foreach(_ -> push!(Bshuftup.args, 0), 1:Wh); foreach(_ -> push!(Bshuftup.args, 1), 1:Wh)
#     Ashuffle = Expr(:call, Expr(:curly, :Val, Ashuftup))
#     Bshuffle = Expr(:call, Expr(:curly, :Val, Bshuftup))
#     MindA = Expr(:call, Expr(:curly, :_MM, Wh), 0)
#     MindC = Expr(:call, Expr(:curly, :_MM, W), 0)
#     for n ∈ 0:Nrep-1
#         for k ∈ 0:Krep-1
#             two_cols_per_vector_add_kn_block!(q, Kunroll, Nunroll, Kunroll*k, Nunroll*n + Noffbase, Ashuffle, Bshuffle, MindA, MindC, Amask)
#         end
#         if Krem > 0
#             two_cols_per_vector_add_kn_block!(q, Krem, Nunroll, Kunroll*Krep, Nunroll*n + Noffbase, Ashuffle, Bshuffle, MindA, MindC, Amask)
#         end
#     end
#     if Nrem > 0
#         for k in 0:Krep-1
#             two_cols_per_vector_add_kn_block!(q, Kunroll, Nrem, Kunroll*k, Nunroll*Nrep + Noffbase, Ashuffle, Bshuffle, MindA, MindC, Amask)
#         end
#         if Krem > 0
#             two_cols_per_vector_add_kn_block!(q, Krem, Nrem, Kunroll*Krep, Nunroll*Nrep + Noffbase, Ashuffle, Bshuffle, MindA, MindC, Amask)
#         end        
#     end
#     if isodd(N)
#         Csym = Symbol(:C_,0)
#         for k ∈ 0:K-1
#             Asym = Symbol(:A_,k)
#             Bsym = Symbol(:B_,k)
#             Aload = Expr(:call, :vload, :ptrA, Expr(:tuple, MindA, k))
#             isnothing(Amask) || push!(Aload.args, Amask)
#             push!(q.args, Expr(:(=), Asym, Expr(:call, :extract_data, Aload)))
#             push!(q.args, Expr(:(=), Bsym, Expr(:call, :vload, :ptrB, Expr(:tuple, k, N-1))))
#             if iszero(k)
#                 push!(q.args, Expr(:(=), Csym, Expr(:call, :vmul_fast, Asym, Bsym)))
#             else
#                 push!(q.args, Expr(:(=), Csym, Expr(:call, :vfmadd, Asym, Bsym, Csym)))
#             end
#         end
#         push!(q.args, Expr(:call, :vnoaliasstore!, :ptrC, Csym, Expr(:tuple, MindA, N-1))) #MindA, because we want half-vector
#     end
#     q
# end

# # function LinearAlgebra.mul!(
# # @generated function LinearAlgebra.mul!(
# #     C::AbstractFixedSizeMatrix{M,N,T,1,4,false},
# #     A::AbstractFixedSizeMatrix{M,K,T,1,XA},
# #     B::AbstractFixedSizeMatrix{K,N,T}
# # ) where {M,K,N,XA,T}
# # # ) where {M,K,N,T,XA}
# #     W, Wshift = VectorizationBase.pick_vector_width_shift(4N, T)
# #     nvectors_per_col = W >> 2
# #     # if nvectors_per_col ≤ 1 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 4
# #     if nvectors_per_col != 2 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 4
# #         return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
# #     elseif M == 3#if nvectors_per_col == 2
# #         return two_cols_per_vector_quote(K, N, W, Wshift, 0x77)
# #     else
# #         return two_cols_per_vector_quote(K, N, W, Wshift)
# #     # else
# #         # return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
# #     end
# # end


# function four_cols_per_vector_add_kn_block!(q::Expr, Kunroll, Nunroll, Koffset, Noffset, Ashuffle, Bshuffle, MindA, MindC)
#     for kk ∈ 0:Kunroll-1
#         kt = kk + Koffset
#         push!(q.args, Expr(:(=), Symbol(:A_,kk), Expr(:call, :shufflevector, Expr(:call, :extract_data, Expr(:call, :vload, :ptrA, Expr(:tuple, MindA, kt))), Ashuffle)))
#     end
#     for kk ∈ 0:Kunroll-1, nn ∈ 0:Nunroll-1
#         kt = kk + Koffset
#         nt1 = 4(nn + Noffset)
#         nt2 = nt1 + 1
#         Bsym = Symbol(:B_,kk,:_,nn)
#         Csym = Symbol(:C_, nn)
#         # push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt)), Bshuffle)))
#         shuffle = Expr(
#             :tuple,
#             Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt1))),
#             Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2))),
#             Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2+1))),
#             Expr(:call, :(Core.VecElement), Expr(:call, :vload, :ptrB, Expr(:tuple, kt, nt2+2)))
#         )
#         push!(q.args, Expr(:(=), Bsym, Expr(:call, :shufflevector, shuffle, Bshuffle)))
#         if iszero(Koffset | kk)
#             push!(q.args, Expr(:(=), Csym, Expr(:call, :vmul_fast, Symbol(:A_, kk), Bsym)))
#         else
#             push!(q.args, Expr(:(=), Csym, Expr(:call, :vfmadd, Symbol(:A_, kk), Bsym, Csym)))
#         end
#     end
#     for nn ∈ 0:Nunroll-1
#         push!(q.args, Expr(:call, :vnoaliasstore!, :ptrC, Symbol(:C_, nn), Expr(:tuple, MindC, 4(nn + Noffset) )))
#     end
# end
# function four_cols_per_vector_quote(K, N, W, Wshift)
#     # Calculate 2 columns of C at a time
#     Nquadreps = N >> 2
#     Nquadrem = N & 3
#     # Nisodd = N & 1
#     q = initmulquote()
#     Wq = W >>> 2
#     Wqs = Wshift - 2

#     Kloop = LoopVectorization.Loop(:k, 1, K, Symbol(""), Symbol(""), true, true)
#     Nloop = LoopVectorization.Loop(:n, 1, Nquadreps, Symbol(""), Symbol(""), true, true)
#     cost_vec     = Float64[ 64.0, 32.0, 8.0, 0.0 ] # Values chosen to hopefully produce desired behavior... TODO: do this better?
#     reg_pressure = Float64[ 1.0, 1.0, 1.0, 0.0, 0.0 ]
#     Kunroll, Nunroll, cost = LoopVectorization.solve_unroll(
#         :k, :n, cost_vec, reg_pressure, W, :m, Kloop, Nloop
#     )
#     @assert isfinite(cost)
    
#     Nrep, Nrem = divrem(Nquadreps, Nunroll)
#     Krep, Krem = divrem(K, Kunroll)
#     Ashuftup = Expr(:tuple);
#     append!(Ashuftup.args, 0:Wq-1); append!(Ashuftup.args, 0:Wq-1)
#     append!(Ashuftup.args, 0:Wq-1); append!(Ashuftup.args, 0:Wq-1)
#     Bshuftup = Expr(:tuple);
#     foreach(_ -> push!(Bshuftup.args, 0), 1:Wq); foreach(_ -> push!(Bshuftup.args, 1), 1:Wq)
#     foreach(_ -> push!(Bshuftup.args, 0), 1:Wq); foreach(_ -> push!(Bshuftup.args, 1), 1:Wq)
#     Ashuffle = Expr(:call, Expr(:curly, :Val, Ashuftup))
#     Bshuffle = Expr(:call, Expr(:curly, :Val, Bshuftup))
#     MindA = Expr(:call, Expr(:curly, :_MM, Wq), 0)
#     MindC = Expr(:call, Expr(:curly, :_MM, W), 0)
#     for n ∈ 0:Nrep-1
#         for k ∈ 0:Krep-1
#             four_cols_per_vector_add_kn_block!(q, Kunroll, Nunroll, Kunroll*k, Nunroll*n, Ashuffle, Bshuffle, MindA, MindC)
#         end
#         if Krem > 0
#             four_cols_per_vector_add_kn_block!(q, Krem, Nunroll, Kunroll*Krep, Nunroll*n, Ashuffle, Bshuffle, MindA, MindC)
#         end
#     end
#     if Nrem > 0
#         for k in 0:Krep-1
#             four_cols_per_vector_add_kn_block!(q, Kunroll, Nrem, Kunroll*k, Nunroll*Nrep, Ashuffle, Bshuffle, MindA, MindC)
#         end
#         if Krem > 0
#             four_cols_per_vector_add_kn_block!(q, Krem, Nrem, Kunroll*Krep, Nunroll*Nrep, Ashuffle, Bshuffle, MindA, MindC)
#         end
#     end
#     if Nquadrem > 0
#         two_cols_per_vector_quote!(q, K, N, W >> 1, Wshift - 1, N & -4)
#     end
#     push!(q.args, :C)
#     q
# end

# function LinearAlgebra.mul!(
# @generated function LinearAlgebra.mul!(
#     C::AbstractFixedSizeMatrix{M,N,T,1,2,false},
#     A::AbstractFixedSizeMatrix{M,K,T,1,XA},
#     B::AbstractFixedSizeMatrix{K,N,T}
# ) where {M,K,N,XA,T}
# # ) where {M,K,N,T,XA}
#     W, Wshift = VectorizationBase.pick_vector_width_shift(2N, T)
#     nvectors_per_col = W >> 1
#     if nvectors_per_col ≤ 1 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 2
#     # if nvectors_per_col != 2 || (K * N > 4VectorizationBase.REGISTER_COUNT) || XA < 4
#         return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
#     elseif nvectors_per_col == 2
#         return two_cols_per_vector_quote(K, N, W, Wshift)
#     # elseif nvectors_per_col == 4
#         # return four_cols_per_vector_quote(K, N, W, Wshift)
#     else
#         return Expr(:block, Expr(:meta,:inline), Expr(:call, :jmul!, :C, :A, :B))
#     end
# end
