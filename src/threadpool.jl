
struct MatMulTask
    problem::Base.RefValue{NTuple{24,UInt}}
    MatMulTask() = new(Base.RefValue{NTuple{24,UInt}}())
end
const MATMULLERS = ntuple(_ -> MatMulTask(), Val(NUM_CORES))

# function loadnonzero(p::Ptr{T}) where {T}
#     z = zero(T)
#     while true
#         u = _atomic_and!(ptr, z)
#         (u !== z) && break
#         pause()
#     end
#     u
# end
Base.pointer(mm::MatMulTask) = Base.unsafe_convert(Ptr{UInt}, pointer_from_objref(mm.problem))

# const MULBUFFER = Base.RefValue{NTuple{20,UInt}}

const SPIN = UInt(0)   # 0: spinning
const WAIT = UInt(1)   # 1: waitin, check if ≤ 1 to see if task is free and waiting
const TASK = UInt(2)   # 2: task available
const LOCK = UInt(3)   # 3: lock
const STUP = UInt(4)   # 4: problem being setup. Any reason to have two lock flags?

function (m::MatMulTask)()
    p = pointer(m)
    problem = m.problem
    max_wait = 1 << 20
    wait_counter = max_wait
    GC.@preserve problem begin
        while true
            if _atomic_cas_cmp!(p, TASK, LOCK)
                _matmul!(p)
                @assert _atomic_cas_cmp!(p, LOCK, SPIN)
                wait_counter = 0
                continue
            end
            pause()
            if (wait_counter += 1) > max_wait
                wait_counter = 0
                if _atomic_cas_cmp!(p, SPIN, WAIT)
                    wait()
                    _matmul!(p)
                    @assert _atomic_cas_cmp!(p, LOCK, SPIN)
                end
            end
        end
    end
end
function launch_thread_mul!(C, A, B, α, β, M, K, N, tid::Int, ::Val{P}) where {P}
    # tid == NUM_CORES && return call_loopmul!(C, A, B, α, β, M, K, N, Val{P}())
    p = pointer(MATMULLERS[tid])
    while true
        if _atomic_cas_cmp!(p, SPIN, STUP)
            setup_matmul!(p, C, A, B, α, β, M, K, N, Val{P}())
            @assert _atomic_cas_cmp!(p, STUP, TASK)
            return
        elseif _atomic_cas_cmp!(p, WAIT, STUP)
            setup_matmul!(p, C, A, B, α, β, M, K, N, Val{P}())
            _atomic_cas_cmp!(p, STUP, LOCK)
            wake_thread!(tid % UInt)
            return
        end
        pause()
    end
end
function launch_thread_mul!(
    C, A, B, α, β, M, K, N, ap, bcp, tid, tt,::StaticFloat{W₁},::StaticFloat{W₂},::StaticFloat{R₁},::StaticFloat{R₂}
) where {W₁,W₂,R₁,R₂}
    p = pointer(MATMULLERS[tid])
    while true
        if _atomic_cas_cmp!(p, SPIN, STUP)
            setup_syncmul!(
                p, C, A, B, α, β, M, K, N, ap, bcp, tid, tt,
                StaticFloat{W₁}(),StaticFloat{W₂}(),StaticFloat{R₁}(),StaticFloat{R₂}()
            )
            @assert _atomic_cas_cmp!(p, STUP, TASK)
            return
        elseif _atomic_cas_cmp!(p, WAIT, STUP)
            # we immediately write the `atomicp, bc, tid, total_tids` part, so we can dispatch to the same code as the other methods
            setup_syncmul!(
                p, C, A, B, α, β, M, K, N, ap, bcp, tid, tt,
                StaticFloat{W₁}(),StaticFloat{W₂}(),StaticFloat{R₁}(),StaticFloat{R₂}()
            )
            _atomic_cas_cmp!(p, STUP, LOCK)
            wake_thread!(tid)
            return
        end
        pause()
    end
end


# 1-based tid
function wake_thread!(tid)
    push!(Base.Workqueues[tid+1], MULTASKS[tid]);
    # push!(@inbounds(Base.Workqueues[tid+1]), MULTASKS[tid]);
    ccall(:jl_wakeup_thread, Cvoid, (Int16,), tid % Int16)
end

# 1-based tid
@inline function __wait(tid::Int)
    p = pointer(MATMULLERS[tid])
    while _atomic_max!(p, SPIN) > WAIT # WARNING: based on relative values
        pause()
    end
end

