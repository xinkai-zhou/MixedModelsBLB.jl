

"""
update_w!(m, w)
Update the weight vector `m.w` using `w`.
"""
function update_w!(
    m::blblmmModel{T}, 
    w::Vector{Int64}
    ) where T <: BlasReal
    copyto!(m.w, w)
end

"""
    init_ls!(m::blblmmModel)

Initialize parameters of a `blblmmModel` object from the least squares estimate. 
`m.β`, `m.ΣL`, and `m.σ²` are overwritten with the least squares estimates.
"""
function init_ls!(m::blblmmModel{T}, verbose::Bool = false) where T <: BlasReal
    # p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    # LS estimate for β
    fill!(m.xtx, 0)
    fill!(m.xty, 0)
    @views @inbounds for i in eachindex(m.data)
        # obs = m.data[i]
        BLAS.axpy!(T(1), m.data[i].xtx, m.xtx)
        BLAS.axpy!(T(1), m.data[i].xty, m.xty)
    end
    LinearAlgebra.copytri!(m.xtx, 'U')
    LAPACK.potrf!('U', m.xtx)
    BLAS.trsv!('U', 'T', 'N', m.xtx, copyto!(m.β, m.xty))
    BLAS.trsv!('U', 'N', 'N', m.xtx, m.β)
    # LS etimate for σ2 and Σ
    rss, ntotal = zero(T), 0
    fill!(m.ztz2, 0)
    fill!(m.ztr2, 0)    
    @inbounds for i in eachindex(m.data)
        obs = m.data[i]
        ntotal += length(obs.y)
        # update Xt * res
        BLAS.gemv!('N', T(-1), obs.xtx, m.β, T(1), copyto!(obs.xtr, obs.xty))
        # rss of i-th individual
        rss += obs.yty[1] - dot(obs.xty, m.β) - dot(obs.xtr, m.β)
        # update Zi' * res
        BLAS.gemv!('N', T(-1), obs.ztx, m.β, T(1), copyto!(obs.ztr, obs.zty))
        # Zi'Zi ⊗ Zi'Zi
        kron_axpy!(obs.ztz, obs.ztz, m.ztz2)
        # Zi'res ⊗ Zi'res
        kron_axpy!(obs.ztr, obs.ztr, m.ztr2)
    end
    m.σ²[1] = rss / ntotal
    # LS estimate for Σ = LLt 
    LinearAlgebra.copytri!(m.ztz2, 'U')
    LAPACK.potrf!('U', m.ztz2)
    BLAS.trsv!('U', 'T', 'N', m.ztz2, copyto!(vec(m.Σ), m.ztr2))
    BLAS.trsv!('U', 'N', 'N', m.ztz2, vec(m.Σ))
    LinearAlgebra.copytri!(m.Σ, 'U')
    verbose && print("in init, m.Σ = ", m.Σ, "\n")
    # ldiv!(vec(m.ΣL), cholesky!(Symmetric(m.ztz2)), m.ztr2)
    copyto!(m.ΣL, m.Σ)
    LAPACK.potrf!('L', m.ΣL)
    for j in 2:m.q, i in 1:j-1
        m.ΣL[i, j] = 0
    end
    # verbose && print("in init, m.ΣL = ", m.ΣL, "\n")
    # mul!(m.Σ, m.ΣL, transpose(m.ΣL))
    # if !isposdef(m.Σ)
    #     copyto!(m.Σ, zeros(m.q, m.q))
    #     for i in 1:m.q
    #         m.Σ[i, i] = 1
    #     end
    # end
    m
end


"""
loglikelihood!(obs::blblmmObs, β::Vector, σ²::Vector, ΣL::Matrix, needgrad::Bool, needhess::Bool)

Evaluate the log-likelihood of the cluster `obs`.
"""
function loglikelihood!(
    obs::blblmmObs{T},
    β::Vector{T},
    σ²::Vector{T}, 
    # Σ::Matrix{T},
    ΣL::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false

    # obs::blblmmObs{T},
    # β::Vector{T},
    # σ²::Vector{T}, 
    # # Σ::Matrix{T},
    # L::Matrix{T},
    # needgrad::Bool = false,
    # needhess::Bool = false
    ) where T <: BlasReal

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    σ²inv = 1 / σ²[1]

    ###########
    # objective
    ###########
    copyto!(obs.storage_qq_1, obs.ztz) 
    # storage_qq_1 = L'Z'Z
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
    needgrad && copyto!(obs.∇L, obs.storage_qq_1)
    # storage_qq_1 = L'Z'Z L
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
    # storage_qq_2 = L'Z'Z L
    needgrad && copyto!(obs.storage_qq_2, obs.storage_qq_1)
    # form σ²I + L'Z'Z L
    @inbounds for i in 1:q
        obs.storage_qq_1[i, i] += σ²[1]
    end
    # storage_qq_1 = upper cholesky factor of σ²I + L'Z'Z L
    LAPACK.potrf!('U', obs.storage_qq_1) 
    # ∇L = chol^{-1} L'Z'Z
    needgrad && BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq_1, obs.∇L)
    if needhess
        # storage_qp = L'Z'X
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, copyto!(obs.storage_qp, obs.ztx))
        # storage_qp = chol^{-1} L'Z'X
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq_1, obs.storage_qp)
    end
    # calculate rtr as yty - β'xty - β'xtr (reason: we will need xtr in ∇β)
    # first calculate xtr
    BLAS.gemv!('N', T(-1), obs.xtx, β, T(1), copyto!(obs.xtr, obs.xty))
    rtr = obs.yty[1] - dot(β, obs.xty) - dot(β, obs.xtr)
    # ztr = Z'r = -Z'Xβ + Z'y
    BLAS.gemv!('N', T(-1), obs.ztx, β, T(1), copyto!(obs.ztr, obs.zty))
    # storage_q_1 = L'Z'r
    BLAS.trmv!('L', 'T', 'N', ΣL, copyto!(obs.storage_q_1, obs.ztr))
    # storage_q_1 = chol^{-1} L'Z'r
    BLAS.trsv!('U', 'T', 'N', obs.storage_qq_1, obs.storage_q_1)
    # calculate the loglikelihood
    logl = n * log(2π) + (n-q) * log(σ²[1])
    @inbounds for i in 1:q
        # logl += 2 * log(obs.storage_qq_1[i, i])
        # the diag of chol may be <=0 due to numerical reasons. 
        # if this happens, set logl to be -Inf.
        if obs.storage_qq_1[i, i] <= 0
            logl = -Inf
            return logl
        else 
            logl += 2 * log(obs.storage_qq_1[i, i])    
        end
    end
    # the quadratic form will be used in grad too
    qf = dot(obs.storage_q_1, obs.storage_q_1)
    logl += σ²inv * (rtr - qf)
    logl /= -2
    obs.obj[1] = logl

    ###########
    # gradient
    ###########
    if needgrad
        # caluclate storage_qq_1 = (σ²I + L'Z'Z L)^{-1} 
        LAPACK.potri!('U', obs.storage_qq_1)
        LinearAlgebra.copytri!(obs.storage_qq_1, 'U') 
        # calculate tr_minvltztzl, which will be used in ∇σ². storage_qq_2 = L'Z'Z L
        tr_minvltztzl = dot(obs.storage_qq_1, obs.storage_qq_2)
        # calculate storage_qq_1 = LMinvL', which will be used repeatedly
        BLAS.trmm!('L', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
        BLAS.trmm!('R', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
        
        # ∇β
        # then calculate storage_q_1 = LM^{-1}L'Z'r
        BLAS.gemv!('N', T(1), obs.storage_qq_1, obs.ztr, T(0), obs.storage_q_1)
        # calculate ∇β
        BLAS.gemv!('T', T(-1), obs.ztx, obs.storage_q_1, T(1), copyto!(obs.∇β, obs.xtr))
        obs.∇β .*= σ²inv

        # ∇σ²
        obs.∇σ²[1] = -σ²inv * (n - tr_minvltztzl)
        # storage_q_2 = Z'ZLM^{-1}L'Z'r
        BLAS.gemv!('N', T(1), obs.ztz, obs.storage_q_1, T(0), obs.storage_q_2)
        obs.∇σ²[1] += abs2(σ²inv) * (rtr - 2 * qf + dot(obs.storage_q_1, obs.storage_q_2))
        obs.∇σ²[1] /= 2

        # ∇Σ (get ∇Σ on the obs level, then get ∇L on the model level)
        # currently ∇L = chol^{-1} L'Z'Z
        # calculate storage_qq_2 = Z'ZLMinvL'Z'Z using a rank-k update
        BLAS.syrk!('U', 'T', T(1), obs.∇L, T(0), obs.storage_qq_2)
        LinearAlgebra.copytri!(obs.storage_qq_2, 'U')
        if needhess
            # Hσ²L
            # currently storage_qq_1 = LMinvL', storage_qq_2 = Z'ZLMinvL'Z'Z
            tr_lminvltztzlminvltztz = dot(obs.storage_qq_1, obs.storage_qq_2)
            # storage_qq_3 = LMinvL'Z'Z
            mul!(obs.storage_qq_3, obs.storage_qq_1, obs.ztz)
            # storage_qq_1 = Z'ZLMinvL'Z'Z LMinvL'Z'Z
            mul!(obs.storage_qq_1, obs.storage_qq_2, obs.storage_qq_3)
            BLAS.axpy!(T(1), obs.ztz, obs.storage_qq_1)
            BLAS.axpy!(T(-2), obs.storage_qq_2, obs.storage_qq_1)
            lmul!(abs2(σ²inv), obs.storage_qq_1)
            # now storage_qq_1 = Z'Ω^{-2}Z, right multiply L to get Z'Ω^{-2}ZL
            BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
            # print("storage_qq_1 = ", obs.storage_qq_1, "\n")
            # D = [1 0 0; 0 1 0; 0 0 0; 0 0 1] 
            # obs.Hσ²L .= vec(vec(obs.storage_qq_1)' * D)
            vech!(obs.Hσ²L, obs.storage_qq_1)
        end
        # calculate storage_qq_2 = -(Z'Z - Z'ZLMinvL'Z'Z)
        BLAS.axpy!(T(-1), obs.ztz, obs.storage_qq_2)
        # storage_qq_2 = -Z'ΩinvZ = -σ²inv * (Z'Z - Z'ZLMinvL'Z'Z)
        lmul!(σ²inv, obs.storage_qq_2)
        copyto!(obs.∇L, obs.storage_qq_2)
        if needhess
            # HLL
            fill!(obs.HLL, T(0))
            # let storage_qq_2 = Z'ΩinvZ
            lmul!(T(-1), obs.storage_qq_2)
            LinearAlgebra.copytri!(obs.storage_qq_2, 'U')
            copyto!(obs.storage_qq_1, obs.storage_qq_2)
            # let storage_qq_1 = Z'ΩinvZL
            BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
            Ct_At_kron_A_KC!(obs.HLL, obs.storage_qq_1)
            # storage_qq_1 = L'Z'ΩinvZL, storage_qq_2 = Z'ΩinvZ
            BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
            Ct_A_kron_B_C!(obs.HLL, obs.storage_qq_1, obs.storage_qq_2)
        end
        # currently storage_q_2 = Z'ZLM^{-1}L'Z'r
        # calculate storage_q_2 = Z'ZLM^{-1}L'Z'r - Z'r 
        BLAS.axpy!(T(-1), obs.ztr, obs.storage_q_2)
        # ∇L = -Z'ΩinvZ + Z'Ωinvrr'ΩinvZ
        BLAS.syr!('U', abs2(σ²inv), obs.storage_q_2, obs.∇L)
        # update ∇L
        LinearAlgebra.copytri!(obs.∇L, 'U')
    end
    ###########
    # hessian
    ###########
    if needhess
        # Hββ
        BLAS.syrk!('U', 'T', T(-1), obs.storage_qp, T(1), copyto!(obs.Hββ, obs.xtx))
        lmul!(σ²inv, obs.Hββ)

        # Hσ²σ²
        obs.Hσ²σ²[1] = (abs2(σ²inv) * (n - 2 * tr_minvltztzl + tr_lminvltztzlminvltztz)) / 2

        # Hσ²L
        # done above

        # HLL
        # done above
    end
    logl

    # n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    # σ²inv = 1 / σ²[1] #inv(σ²)
    # ####################
    # # Evaluate objective
    # ####################
    # # form the q-by-q matrix: Lt Zt Z L
    # copy!(obs.ltztzl, obs.ztz)
    # BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.ltztzl) # O(q^3)
    # BLAS.trmm!('R', 'L', 'N', 'N', T(1), L, obs.ltztzl) # O(q^3)    
    # # form the q-by-q matrix: M = σ² I + Lt Zt Z L
    # copy!(obs.M, obs.ltztzl)
    # @inbounds for j in 1:q
    #     obs.M[j, j] += σ²[1]
    # end
    # # cholesky on M = σ² I + Lt Zt Z L
    # LAPACK.potrf!('U', obs.M) # O(q^3)
    # # Zt * res
    # BLAS.gemv!('N', T(-1), obs.ztx, β, T(1), copy!(obs.ztr, obs.zty)) # O(pq)
    # # Lt * Zt * res
    # BLAS.trmv!('L', 'T', 'N', L, copy!(obs.ltztr, obs.ztr))    # O(q^2)
    # # storage_q = (Mchol.U') \ (Lt * (Zt * res))
    # BLAS.trsv!('U', 'T', 'N', obs.M, copy!(obs.storage_q, obs.ltztr)) # O(q^3)
    # # Xt * res = Xt * y - Xt * X * β
    # BLAS.gemv!('N', T(-1), obs.xtx, β, T(1), copy!(obs.xtr, obs.xty))
    # # l2 norm of residual vector
    # rtr = obs.yty - dot(obs.xty, β) - dot(obs.xtr, β)
    # # assemble pieces
    # logl::T = n * log(2π) + (n - q) * log(σ²[1]) # constant term
    # @inbounds for j in 1:q # log det term
    #     logl += 2log(obs.M[j, j])
    # end
    # qf    = abs2(norm(obs.storage_q)) # quadratic form term
    # logl += (rtr - qf) * σ²inv
    # logl /= -2
    # ###################
    # # Evaluate gradient
    # ###################    
    # if needgrad
    #     # M is Minv now: M = (σ² I + Lt Zt Z L)^{-1}
    #     Minv = LAPACK.potri!('U', obs.M)
    #     LinearAlgebra.copytri!(Minv, 'U')
    #     # form lminvlt = L * Minv * Lt
    #     copy!(obs.lminvlt, Minv)
    #     BLAS.trmm!('L', 'L', 'N', 'N', T(1), L, obs.lminvlt) # O(q^3)
    #     BLAS.trmm!('R', 'L', 'T', 'N', T(1), L, obs.lminvlt) # O(q^3)          
    #     # storage_q = (L * Minv * Lt) * (Zt * res)
    #     BLAS.gemv!('N', T(1), obs.lminvlt, obs.ztr, T(0), obs.storage_q)
    #     # compute ∇β
    #     BLAS.gemv!('T', T(-1), obs.ztx, obs.storage_q, T(1), copy!(obs.∇β, obs.xtr))
    #     obs.∇β .*= σ²inv
    #     # compute ∇σ²        
    #     BLAS.gemv!('N', T(1), obs.ztz, obs.storage_q, T(0), obs.ltztr)
    #     obs.∇σ²[1]  = (rtr - 2qf + dot(obs.storage_q, obs.ltztr)) * abs2(σ²inv)
    #     tr_minvltztzl = dot(obs.ltztzl, Minv) # needed in hessian too
    #     obs.∇σ²[1] -= (n - tr_minvltztzl) * σ²inv
    #     obs.∇σ²[1] /= 2
    #     # compute ∇Σ
    #     # Form ztΩinvz = (obs.ztz .- obs.ztz * obs.lminvlt * obs.ztz) / σ²
    #     mul!(obs.∇Σ, obs.ztz, obs.lminvlt)
    #     mul!(obs.ztΩinvz, obs.∇Σ, obs.ztz)
    #     if needhess
    #         tr_lminvltztzlminvltztz = dot(obs.lminvlt, obs.ztΩinvz) # to be used later
    #         # calcualte obs.Hσ²L
    #         copy!(obs.Hσ²L, obs.ztΩinvz)
    #         BLAS.gemm!('N', 'N', T(1), obs.∇Σ, obs.ztΩinvz, T(-2), obs.Hσ²L)
    #         obs.Hσ²L .+= obs.ztz
    #         obs.Hσ²L .*= -abs2(σ²inv)
    #         vech!(obs.Hσ²Lvec, obs.Hσ²L)
    #     end
    #     obs.ztΩinvz .= (obs.ztz .- obs.ztΩinvz) .* σ²inv
    #     # obs.∇Σ .= - obs.ztΩinvz + (Zt * inv(Ω) * res) * (Zt * inv(Ω) * res)^T
    #     copy!(obs.∇Σ, obs.ztΩinvz)
    #     # obs.ztr = Zt * inv(Ω) * res now
    #     BLAS.gemv!('N', T(-1), obs.ztz, obs.storage_q, T(1), obs.ztr)
    #     obs.ztr .*= σ²inv
    #     # assemble pieces
    #     BLAS.syrk!('U', 'N', T(1), obs.ztr, T(-1), obs.∇Σ)
    #     LinearAlgebra.copytri!(obs.∇Σ, 'U')
    # end
    # ###################
    # # Evaluate Hessian
    # ###################    
    # if needhess
    #     # Hββ = - σ²inv * (xtx - xtz * (L * Minv * Lt) * ztx)
    #     mul!(obs.storage_pq, transpose(obs.ztx), obs.lminvlt)
    #     mul!(obs.Hββ, obs.storage_pq, obs.ztx)
    #     obs.Hββ .= (obs.Hββ .- obs.xtx) .* σ²inv
    #     # HLL
    #     fill!(obs.HLL, 0)
    #     # storage_qq = (Zt * Ωinv * Z) * L
    #     mul!(obs.storage_qq, obs.ztΩinvz, L)
    #     Ct_At_kron_A_KC!(obs.HLL, obs.storage_qq)
    #     # storage_qq = Lt * (Zt * Ωinv * Z) * L
    #     BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.storage_qq)
    #     Ct_A_kron_B_C!(obs.HLL, obs.storage_qq, obs.ztΩinvz)
    #     obs.HLL .*= -1
    #     # Hσ²σ²
    #     obs.Hσ²σ²[1, 1] = - (n - 2tr_minvltztzl + tr_lminvltztzlminvltztz) * abs2(σ²inv) / 2
    #     # Hσ²L was done earlier in gradient calculation
    # end
    # ###################
    # # Return
    # ###################        
    # return logl
end

# function update_logl_multithreaded!(
#     m::blblmmModel{T}, 
#     needgrad::Bool = false, 
#     needhess::Bool = false
#     ) where T <: BlasReal
#     Threads.@threads for obs in m.data  
#         loglikelihood!(obs, m.β, m.σ², m.ΣL, needgrad, needhess)
#     end
# end

function loglikelihood!(
    m::blblmmModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    logl = zero(T)
    if needgrad
        fill!(m.∇β, 0)
        fill!(m.∇σ², 0)
        fill!(m.∇L, 0)
    end
    if needhess
        fill!(m.Hββ, 0)
        fill!(m.Hσ²σ², 0)
        fill!(m.HLL, 0)
        fill!(m.Hσ²L, 0)
    end
    if m.use_threads
        # update_logl_multithreaded!(m, needgrad, needhess)
        Threads.@threads for obs in m.data  
            loglikelihood!(obs, m.β, m.σ², m.ΣL, needgrad, needhess)
        end
        # let
        #     Threads.@threads for obs in m.data  
        #         loglikelihood!(obs, m.β, m.σ², m.ΣL, needgrad, needhess)
        #     end
        # end
        @inbounds for i in eachindex(m.data)
            logl += m.w[i] * m.data[i].obj[1]
            if needgrad
                # obs = m.data[i]
                BLAS.axpy!(m.w[i], m.data[i].∇β, m.∇β)
                m.∇σ²[1] += m.w[i] * m.data[i].∇σ²[1]
                BLAS.axpy!(m.w[i], m.data[i].∇L, m.∇L)
            end
            if needhess
                # obs = m.data[i]
                BLAS.axpy!(m.w[i], m.data[i].Hββ, m.Hββ)
                m.Hσ²σ²[1] += m.w[i] * m.data[i].Hσ²σ²[1]
                BLAS.axpy!(m.w[i], m.data[i].HLL, m.HLL)
                BLAS.axpy!(m.w[i], m.data[i].Hσ²L, m.Hσ²L)
            end
        end
    else 
        @inbounds for i in eachindex(m.data)
            logl += m.w[i] * loglikelihood!(m.data[i], m.β, m.σ², m.ΣL, needgrad, needhess)
            if needgrad
                # obs = m.data[i]
                BLAS.axpy!(m.w[i], m.data[i].∇β, m.∇β)
                m.∇σ²[1] += m.w[i] * m.data[i].∇σ²[1]
                BLAS.axpy!(m.w[i], m.data[i].∇L, m.∇L)
            end
            if needhess
                # obs = m.data[i]
                BLAS.axpy!(m.w[i], m.data[i].Hββ, m.Hββ)
                m.Hσ²σ²[1] += m.w[i] * m.data[i].Hσ²σ²[1]
                BLAS.axpy!(m.w[i], m.data[i].HLL, m.HLL)
                BLAS.axpy!(m.w[i], m.data[i].Hσ²L, m.Hσ²L)
            end
        end
    end
    # To save cost, we didn't multiply ΣL above in the expression of ∇L
    # Here we do the multiplication
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), m.ΣL, m.∇L)
    logl
end


# function fit!(
#     m::blblmmModel,
#     solver = Ipopt.IpoptSolver(print_level=5)
#     )
#     npar = m.p + 1 + ◺(m.q) #(q * (q + 1)) >> 1
#     # since X includes a column of 1, p is the number of mean parameters
#     # the cholesky factor for the qxq random effect mx has (q * (q + 1)) / 2 values,
#     # the arithmetic shift right operation has the effect of division by 2^n, here n = 1
#     # then there is the error variance

#     # mean effects and intercept (p + 1), random effect covariance (q * q), error variance (1)
#     optm = MathProgBase.NonlinearModel(solver)
#     lb = fill(-Inf, npar) # error variance should be nonnegative, will fix later
#     ub = fill( Inf, npar)
#     MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, m)
#     # starting point
#     par0 = zeros(npar)
#     modelpar_to_optimpar!(par0, m)
#     MathProgBase.setwarmstart!(optm, par0)
#     # optimize
#     MathProgBase.optimize!(optm)
#     # print("after optimize!, getsolution(optm) = ", MathProgBase.getsolution(optm), "\n")
#     optstat = MathProgBase.status(optm)
#     optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
#     # refresh gradient and Hessian
#     optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
#     loglikelihood!(m, true, true) 
#     m
# end

function fit!(
    m::blblmmModel,
    solver = Ipopt.IpoptSolver(print_level=0, warm_start_init_point = "yes", warm_start_bound_push = 1e-9)
    )
    npar = m.p + 1 + ◺(m.q) #(q * (q + 1)) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the qxq random effect mx has (q * (q + 1)) / 2 values,
    # the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    # then there is the error variance

    # mean effects and intercept (p + 1), random effect covariance (q * q), error variance (1)
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar) # error variance should be nonnegative, will fix later
    ub = fill( Inf, npar)
    # σ² should be >= 0
    lb[m.p+1] = 0
    # diag of L >=0
    offset = m.p + 2
    for j in 1:m.q, i in j:m.q
        i == j && (lb[offset] = 0)
        offset += 1
    end
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, m)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, m)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    MathProgBase.optimize!(optm)
    # print("after optimize!, getsolution(optm) = ", MathProgBase.getsolution(optm), "\n")
    optstat = MathProgBase.status(optm)
    # optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    if !(optstat == :Optimal || optstat == :FeasibleApproximate)
        @warn("Optimization unsuccesful; got $optstat")
    end
    # refresh gradient and Hessian
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    loglikelihood!(m, true, true) 
    m
end

# """
#     modelpar_to_optimpar!(m, par)
# Translate model parameters in `m` to optimization variables in `par`.
# """
# function modelpar_to_optimpar!(
#     par::Vector,
#     m::blblmmModel
#     )
#     copyto!(par, m.β)
#     par[m.p+1] = log(m.σ²[1]) # take log and then exp() later to make the problem unconstrained
    
#     copyto!(m.ΣL, Symmetric(m.Σ))
#     LAPACK.potrf!('L', m.ΣL)

#     # # Since modelpar_to_optimpar is only called once, it's ok to allocate Σchol
#     # Σchol = cholesky(Symmetric(m.Σ), Val(false); check = false)
#     # # By using cholesky decomposition and optimizing L, 
#     # # we transform the constrained opt problem (Σ is pd) to an unconstrained problem. 
#     # # m.ΣL .= Σchol.L
#     # copyto!(m.ΣL, Σchol.L)

#     offset = m.p + 2
#     @inbounds for j in 1:m.q
#         par[offset] = log(m.ΣL[j, j]) # only the diagonal is constrained to be nonnegative
#         offset += 1
#         @inbounds for i in j+1:m.q
#             par[offset] = m.ΣL[i, j]
#             offset += 1
#         end
#     end
#     par
# end

"""
    modelpar_to_optimpar!(m, par)
Translate model parameters in `m` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par::Vector,
    m::blblmmModel
    )
    copyto!(par, m.β)
    par[m.p+1] = m.σ²[1] # take log and then exp() later to make the problem unconstrained
    
    copyto!(m.ΣL, Symmetric(m.Σ))
    LAPACK.potrf!('L', m.ΣL)

    # # Since modelpar_to_optimpar is only called once, it's ok to allocate Σchol
    # Σchol = cholesky(Symmetric(m.Σ), Val(false); check = false)
    # # By using cholesky decomposition and optimizing L, 
    # # we transform the constrained opt problem (Σ is pd) to an unconstrained problem. 
    # # m.ΣL .= Σchol.L
    # copyto!(m.ΣL, Σchol.L)

    offset = m.p + 2
    @inbounds for j in 1:m.q
        par[offset] = m.ΣL[j, j] # only the diagonal is constrained to be nonnegative
        offset += 1
        @inbounds for i in j+1:m.q
            par[offset] = m.ΣL[i, j]
            offset += 1
        end
    end
    par
end

# """
#     optimpar_to_modelpar!(m, par)
# Translate optimization variables in `par` to the model parameters in `m`.
# """
# function optimpar_to_modelpar!(
#     m::blblmmModel, 
#     par::Vector)
#     copyto!(m.β, 1, par, 1, m.p)
#     m.σ²[1] = exp(par[m.p+1])
#     fill!(m.ΣL, 0)
#     offset = m.p + 2
#     @inbounds for j in 1:m.q
#         m.ΣL[j, j] = exp(par[offset])
#         offset += 1
#         @inbounds for i in j+1:m.q
#             m.ΣL[i, j] = par[offset]
#             offset += 1
#         end
#     end
#     mul!(m.Σ, m.ΣL, transpose(m.ΣL))
#     m
# end

"""
    optimpar_to_modelpar!(m, par)
Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
    m::blblmmModel, 
    par::Vector)
    copyto!(m.β, 1, par, 1, m.p)
    m.σ²[1] = par[m.p+1]
    fill!(m.ΣL, 0)
    offset = m.p + 2
    @inbounds for j in 1:m.q
        m.ΣL[j, j] = par[offset]
        offset += 1
        @inbounds for i in j+1:m.q
            m.ΣL[i, j] = par[offset]
            offset += 1
        end
    end
    mul!(m.Σ, m.ΣL, transpose(m.ΣL))
    m
end

function MathProgBase.initialize(
    m::blblmmModel, 
    requested_features::Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
        # if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(m::blblmmModel) = [:Grad, :Hess] #[:Grad]

function MathProgBase.eval_f(
    m::blblmmModel, 
    par::Vector)
    optimpar_to_modelpar!(m, par)
    loglikelihood!(m, false, false)
end


# function MathProgBase.eval_grad_f(
#     m::blblmmModel, 
#     grad::Vector, 
#     par::Vector)
#     optimpar_to_modelpar!(m, par)
#     loglikelihood!(m, true, false)
#     # gradient wrt β
#     copyto!(grad, m.∇β)
#     # gradient wrt log(σ²)
#     grad[m.p+1] = m.∇σ²[1] * m.σ²[1]
#     offset = m.p + 2
#     # gradient wrt log(diag(L)) and off-diag(L)
#     @inbounds for j in 1:m.q
#         # On the diagonal, gradient wrt log(ΣL[j,j])
#         grad[offset] = m.∇L[j, j] * m.ΣL[j, j]
#         offset += 1
#         @inbounds for i in j+1:m.q
#             # Off-diagonal, wrt ΣL[i,j]
#             grad[offset] = m.∇L[i, j]
#             offset += 1
#         end
#     end
#     # print("par = ", par, "\n")
#     # print("grad = ", grad, "\n")
#     nothing
# end


function MathProgBase.eval_grad_f(
    m::blblmmModel, 
    grad::Vector, 
    par::Vector)
    optimpar_to_modelpar!(m, par)
    loglikelihood!(m, true, false)
    # gradient wrt β
    copyto!(grad, m.∇β)
    # gradient wrt log(σ²)
    grad[m.p+1] = m.∇σ²[1]
    offset = m.p + 2
    # gradient wrt log(diag(L)) and off-diag(L)
    @inbounds for j in 1:m.q
        # On the diagonal, gradient wrt log(ΣL[j,j])
        grad[offset] = m.∇L[j, j]
        offset += 1
        @inbounds for i in j+1:m.q
            # Off-diagonal, wrt ΣL[i,j]
            grad[offset] = m.∇L[i, j]
            offset += 1
        end
    end
    # print("par = ", par, "\n")
    # print("grad = ", grad, "\n")
    nothing
end

MathProgBase.eval_g(m::blblmmModel, g, par) = nothing
MathProgBase.jac_structure(m::blblmmModel) = Int[], Int[]
MathProgBase.eval_jac_g(m::blblmmModel, J, par) = nothing

function MathProgBase.hesslag_structure(m::blblmmModel)
    # Get the linear indices of the UPPER-triangular of the non-zero blocks
    npar = ◺(m.p) + 1 + ◺(◺(m.q)) + ◺(m.q)
    #       ββ    σ²σ²  σ²vech(L)   vech(L)vech(L) 
    arr1 = Vector{Int}(undef, npar)
    arr2 = Vector{Int}(undef, npar)
    idx = 1
    # Hββ
    for j in 1:m.p
        for i in 1:j
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # Hσ²σ²
    arr1[idx] = m.p + 1
    arr2[idx] = m.p + 1
    idx += 1
    # HLL, take the upper triangle
    for j in (m.p+2):(m.p + 1 + ◺(m.q))
        for i in (m.p+2):j
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # Hσ²L
    for j in (m.p+2):(m.p + 1 + ◺(m.q))
        arr1[idx] = m.p + 1 # same row idx as σ²
        arr2[idx] = j
        idx += 1
    end
    return (arr1, arr2)
end


# function MathProgBase.eval_hesslag(
#     m::blblmmModel, 
#     H::Vector{T},
#     par::Vector{T}, 
#     σ::T, 
#     μ::Vector{T}) where {T}    
#     # l, q◺ = m.l, ◺(m.q)
#     optimpar_to_modelpar!(m, par)
#     # Do we need to evaluate logl here? Since hessian is always evaluated 
#     # after the gradient, can we just evaluate logl once in the gradient step?
#     loglikelihood!(m, true, true)
#     idx = 1
#     @inbounds for j in 1:m.p, i in 1:j
#         H[idx] = m.Hββ[i, j]
#         idx += 1
#     end
#     # hessian wrt log(σ²)
#     H[idx] = m.Hσ²σ²[1] * m.σ²[1]^2
#     idx += 1
    
#     # Since we took log of the diagonal elements, log(ΣL[j,j])
#     # we need to do scaling as follows
#     @inbounds for (iter, icontent) in enumerate(m.diagidx)
#         # On the diagonal we have hessian wrt log(ΣL[j,j])
#         @inbounds for j in 1:m.q◺
#             m.HLL[icontent, j] = m.HLL[icontent, j] * m.ΣL[iter, iter]
#             m.HLL[j, icontent] = m.HLL[j, icontent] * m.ΣL[iter, iter]
#         end
#         m.Hσ²L[icontent] = m.Hσ²L[icontent] * m.ΣL[iter, iter]
#     end

#     @inbounds for j in 1:◺(m.q), i in 1:j
#         H[idx] = m.HLL[i, j] 
#         idx += 1
#     end
#     @inbounds for j in 1:◺(m.q)
#         H[idx] = m.Hσ²L[j] * m.σ²[1]
#         idx += 1
#         # # On the diagonal, wrt log(σ²) and log(ΣL[j,j]) 
#         # H[idx] = m.Hσ²L[j, j] * m.σ²[1]
#         # idx += 1
#         # # Off-diagonal, wrt log(σ²) and ΣL[i,j]
#         # for i in (j+1):m.q
#         #     H[idx] = m.Hσ²L[i, j] * m.σ²[1]
#         #     idx += 1
#         # end
#     end
#     lmul!(T(-1), H)
#     lmul!(σ, H)
# end


function MathProgBase.eval_hesslag(
    m::blblmmModel, 
    H::Vector{T},
    par::Vector{T}, 
    σ::T, 
    μ::Vector{T}) where {T}    
    # l, q◺ = m.l, ◺(m.q)
    optimpar_to_modelpar!(m, par)
    # Do we need to evaluate logl here? Since hessian is always evaluated 
    # after the gradient, can we just evaluate logl once in the gradient step?
    loglikelihood!(m, true, true)
    idx = 1
    @inbounds for j in 1:m.p, i in 1:j
        H[idx] = m.Hββ[i, j]
        idx += 1
    end
    # hessian wrt log(σ²)
    H[idx] = m.Hσ²σ²[1]
    idx += 1
    
    # Since we took log of the diagonal elements, log(ΣL[j,j])
    # we need to do scaling as follows
    # @inbounds for (iter, icontent) in enumerate(m.diagidx)
    #     # On the diagonal we have hessian wrt log(ΣL[j,j])
    #     @inbounds for j in 1:m.q◺
    #         m.HLL[icontent, j] = m.HLL[icontent, j] * m.ΣL[iter, iter]
    #         m.HLL[j, icontent] = m.HLL[j, icontent] * m.ΣL[iter, iter]
    #     end
    #     m.Hσ²L[icontent] = m.Hσ²L[icontent] * m.ΣL[iter, iter]
    # end

    @inbounds for j in 1:◺(m.q), i in 1:j
        H[idx] = m.HLL[i, j] 
        idx += 1
    end
    @inbounds for j in 1:◺(m.q)
        H[idx] = m.Hσ²L[j] 
        idx += 1
        # # On the diagonal, wrt log(σ²) and log(ΣL[j,j]) 
        # H[idx] = m.Hσ²L[j, j] * m.σ²[1]
        # idx += 1
        # # Off-diagonal, wrt log(σ²) and ΣL[i,j]
        # for i in (j+1):m.q
        #     H[idx] = m.Hσ²L[i, j] * m.σ²[1]
        #     idx += 1
        # end
    end
    lmul!(T(-1), H)
    lmul!(σ, H)
end