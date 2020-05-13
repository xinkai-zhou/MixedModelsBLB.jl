


"""
update_res!(obs, β)
Update the residual vector according to `β`.
"""
function update_res!(
    obs::blblmmObs{T}, 
    β::Vector{T}
    ) where T <: BlasReal
    copyto!(obs.res, obs.y)
    BLAS.gemv!('N', -one(T), obs.X, β, one(T), obs.res) # obs.res - obs.X * β = obs.y - obs.X * β
    obs.res
end
function update_res!(
    m::blblmmModel{T}
    ) where T <: BlasReal
    @inbounds for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
    end
    nothing
end

"""
update_w!(m, w)
Update the weight vector using w
"""
function update_w!(
    m::blblmmModel{T}, 
    w::Vector{T}
    ) where T <: BlasReal
    copyto!(m.w, w)
end

# Extract the variance-covariance matrix of variance components
function extract_Σ!(Σ, lmm::LinearMixedModel)
    σρ = MixedModels.VarCorr(lmm).σρ
    q = size(Σ, 1) #length(σρ[1][1])
    @inbounds @views for i in 1:q
        Σ[i, i] = (σρ[1][1][i])^2
        @inbounds for j in (i+1):q
            Σ[i, j] = σρ[1][2][(j-1)] * σρ[1][1][i] * σρ[1][1][j]
        end
    end
    LinearAlgebra.copytri!(Σ, 'U')
    return(Σ)
end

function loglikelihood!(
    obs::blblmmObs{T},
    β::Vector{T},
    σ2::Vector{T}, # inverse of linear regression variance
    Σ::Matrix{T},
    ΣL::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false,
    updateres::Bool = false
    ) where T <: BlasReal

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    σ2inv = 1 / σ2[1]
    if needgrad
        fill!(obs.∇β, T(0))
        fill!(obs.∇σ2, T(0))
        fill!(obs.∇L, T(0))
    end
    if needgrad
        fill!(obs.Hβ, T(0))
        fill!(obs.Hσ2, T(0))
        fill!(obs.HL, T(0))
        fill!(obs.Hσ2L, T(0))
    end
    ###########
    # objective
    ###########
    updateres && update_res!(obs, β)
    Ω = obs.Z * Σ * obs.Z'
    for i in 1:n
        Ω[i, i] += σ2[1]
    end
    Ωinv = inv(Ω)
    logl = n * log(2π) + logdet(Ω) + obs.res' * Ωinv * obs.res
    logl /= -2
    
    # copy!(obs.storage_qq, obs.ztz) 
    # # L'Z'Z
    # BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq)
    # needgrad && copy!(obs.∇L, obs.storage_qq)
    # # σ2 L'Z'Z L
    # BLAS.trmm!('R', 'L', 'N', 'N', σ2[1], ΣL, obs.storage_qq)
    # # storage_qq_1 = σ2 L'Z'Z L
    # needgrad && copy!(obs.storage_qq_1, obs.storage_qq) 
    # # I + L'Z'ZL
    # @inbounds for i in 1:q
    #     obs.storage_qq[i, i] += T(1)
    # end
    # LAPACK.potrf!('U', obs.storage_qq) # cholesky on I + L'Z'ZL
    # # Z'r
    # BLAS.gemv!('T', T(1), obs.Z, obs.res, T(0), obs.storage_q)
    # # L'Z'r
    # BLAS.trmv!('L', 'T', 'N', ΣL, obs.storage_q)
    # # chol^{-1} L'Z'r
    # # BLAS.trsv!(ul, tA, dA, obs.storage_qq, b)
    # BLAS.trsv!('U', 'T', 'N', obs.storage_qq, obs.storage_q)
    # logl = n * log(2π * σ2inv)
    # @inbounds for i in 1:q
    #     if obs.storage_qq[i, i] <= 0
    #         logl = -Inf
    #         return logl
    #     else 
    #         # Calculate logdet(I + L'Z'ZL) through cholesky.
    #         logl += 2 * log(obs.storage_qq[i, i])    
    #     end
    # end
    # logl += σ2[1] * dot(obs.res, obs.res) - σ2[1]^2 * dot(obs.storage_q, obs.storage_q)
    # logl /= -2

    ###########
    # gradient
    ###########
    if needgrad
        obs.∇β .= obs.X' * Ωinv * obs.res
        obs.∇σ2[1] = tr(Ωinv) - obs.res' * Ωinv * Ωinv * obs.res
        obs.∇σ2[1] /= -2
        obs.storage_q .= obs.Z' * Ωinv * obs.res
        obs.∇L .= - obs.Z' * Ωinv * obs.Z +  obs.storage_q * obs.storage_q'
    #     # currently, storage_q = chol^{-1} L'Z'r
    #     # update storage_q to (I+σ2LZ'ZL)^{-1} L'Z'r
    #     BLAS.trsv!('U', 'N', 'N', obs.storage_qq, obs.storage_q)
    #     # update storage_q to L(I+σ2LZ'ZL)^{-1} L'Z'r
    #     BLAS.trmv!('L', 'N', 'N', ΣL, obs.storage_q)

    #     # wrt β
    #     # First calculate σ2X'r
    #     BLAS.gemv!('T', σ2[1], obs.X, obs.res, T(0), obs.∇β)
    #     # then, obs.∇β = σ2X'r - σ2^2 X'Z L(I+σ2LZ'ZL)^{-1} L'Z'r
    #     BLAS.gemv!('T', -σ2[1]^2, obs.ztx, obs.storage_q, T(1), obs.∇β)

    #     # wrt σ2
    #     # Since we no longer need obs.res, update obs.res to be Ω^{-1} r
    #     # σ2r - σ2^2 Z L(I+σ2LZ'ZL)^{-1} L'Z'r
    #     BLAS.gemv!('N', -σ2[1]^2, obs.Z, obs.storage_q, σ2[1], obs.res)
    #     # To evaluate tr(Ω^{-1}), Calculate (I+σ2LZ'ZL)^{-1} L'Z'ZL 
    #     # Currently, storage_qq_1 = σ2 L'Z'Z L, so we multiply 1/σ2[1] (I+σ2LZ'ZL)^{-1}, 
    #     # which is equivalent to two triangular solves
    #     BLAS.trsm!('L', 'U', 'T', 'N', 1/σ2[1], obs.storage_qq, obs.storage_qq_1)
    #     BLAS.trsm!('L', 'U', 'N', 'N', T(1), obs.storage_qq, obs.storage_qq_1)
    #     # print("σ2inv = ", σ2inv, "\n")
    #     # print("tr(obs.storage_qq_1) = ", tr(obs.storage_qq_1), "\n")
    #     if needhess
    #         # calculate the first two terms in (1/2σ2^4) tr(Ω^{-1}Ω^{-1})
    #         obs.Hσ2[1] = (σ2inv^2 / 2) * n - σ2inv^2 * tr(obs.storage_qq_1)
    #     end
    #     obs.∇σ2[1] = (σ2inv / 2) * n - (1//2) * tr(obs.storage_qq_1) - 
    #                 (σ2inv^2 / 2) * dot(obs.res, obs.res)

    #     # Currently, storage_qq is holding the chol factor
    #     # Before it is destroyed, we need to calculate chol^{-1} L'Z'X for the hessian of β
    #     if needhess
    #         copy!(obs.storage_qp, obs.ztx)
    #         # L'Z'X
    #         BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qp)
    #         # chol^{-1} L'Z'X 
    #         BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.storage_qp)
    #     end

    #     # wrt L
    #     # Currently, obs.∇L = L'Z'Z
    #     # Calculate chol^{-1}L'Z'Z and store it in obs.∇L
    #     BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.∇L)
    #     # Currently, storage_qq_1 = (I+σ2LZ'ZL)^{-1} L'Z'ZL 
    #     ## Before destroying storage_qq_1, we need to calculate its product with 
    #     ## itself for the hessian of \tau. Since storage_qq is no longer needed,
    #     ## we overwrite it with the result.
    #     if needhess
    #         BLAS.gemm!('N', 'N', T(1), obs.storage_qq_1, obs.storage_qq_1, T(0), obs.storage_qq)
    #         # print("tr(obs.storage_qq) = ", tr(obs.storage_qq), "\n")
    #         obs.Hσ2[1] += (1//2) * tr(obs.storage_qq)

    #         # Also calculate the hessian for the cross term Hσ2L
    #         # first set Hσ2L = (I+σ2LZ'ZL)^{-1} L'Z'ZL 
    #         copy!(obs.Hσ2L, obs.storage_qq_1)
    #         # then 2σ2(I+σ2LZ'ZL)^{-1} L'Z'ZL - σ2^2(I+σ2LZ'ZL)^{-1} L'Z'ZL(I+σ2LZ'ZL)^{-1} L'Z'ZL 
    #         BLAS.axpby!(-σ2[1]^2, obs.storage_qq, 2*σ2[1], obs.Hσ2L)
    #         # add -I
    #         @inbounds for i in 1:q
    #             obs.Hσ2L[i, i] -= 1
    #         end
    #         # Left multiply by L
    #         BLAS.trmm!('L', 'L', 'N', 'N', T(1), ΣL, obs.Hσ2L)
    #         # The above calculations were all transposed.
    #         # Next we transpose it, and right multiply ztz
    #         BLAS.gemm!('T', 'N', T(1), obs.Hσ2L, obs.ztz, T(0), obs.storage_qq_1)
    #         copy!(obs.Hσ2L, obs.storage_qq_1)
    #         # We should use the lower triangle of Hσ2L
    #     end
    #     # Next we continue with the calculation for ∇L
    #     # Since we no longer need storage_qq_1, we overwrite it with a rank-k update:
    #     # storage_qq_1 = σ2^2 * Z'ZL(I+σ2LZ'ZL)^{-1}L'Z'Z, which is part of Z'Ω^{-1}Z
    #     BLAS.syrk!('U', 'T', σ2[1]^2, obs.∇L, T(0), obs.storage_qq_1) 
    #     # Since syrk only updated the upper triangle, do copytri
    #     LinearAlgebra.copytri!(obs.storage_qq_1, 'U') 
    #     # Update obs.storage_qq_1 as -Z'Ω^{-1}Z = -σ2Z'Z + σ2^2 * Z'ZL(I+σ2LZ'ZL)^{-1}L'Z'Z
    #     BLAS.axpy!(-σ2[1], obs.ztz, obs.storage_qq_1)
    #     # copy it to obs.∇L
    #     copy!(obs.∇L, obs.storage_qq_1)
    #     # Currently res = Ω^{-1} r
    #     # Update storage_q as Z'Ω^{-1} r
    #     BLAS.gemv!('T', T(1), obs.Z, obs.res, T(0), obs.storage_q)
    #     # Rank-1 update of ∇L as ∇L + storage_q storage_q'
    #     BLAS.syr!('U', T(1), obs.storage_q, obs.∇L)
    #     LinearAlgebra.copytri!(obs.∇L, 'U')
    #     # In fact, this is only the gradient wrt Σ
    #     # We will right multiply L once in the aggregation step to save computation
    end

    ###########
    # hessian
    ###########
    if needhess
        obs.Hβ .= obs.X' * Ωinv * obs.X
        obs.Hσ2[1] = tr(Ωinv * Ωinv)
        obs.Hσ2[1] /= -2  
        A = obs.Z' * Ωinv * obs.Z 
        B = A * ΣL
        C = ΣL' * B
        obs.HL .= -Ct_At_kron_A_KC(B) + 3 * Ct_A_kron_B_C(C, A)
        obs.Hσ2L .= ΣL' * obs.Z' * Ωinv * Ωinv * obs.Z
    #     # wrt β
    #     # Currently, storage_qp = chol^{-1} L'Z'X.
    #     copy!(obs.Hβ, obs.xtx)
    #     # Currently, storage_qp = chol^{-1} L'Z'X 
    #     # Update Hβ as -σ2X'X + σ2^2 * X'ZL(I+σ2LZ'ZL)^{-1}L'Z'X through rank-k update
    #     BLAS.syrk!('U', 'T', σ2[1]^2, obs.storage_qp, -σ2[1], obs.Hβ) 
    #     # only the upper triangular of Hβ is updated
        
    #     # wrt σ2
    #     # already calculated above

    #     # wrt vech L 
    #     fill!(obs.HL, T(0))
    #     # Currently storage_qq_1 = -Z'Ω^{-1}Z, and is Symmetric
    #     # We update it to Z'Ω^{-1}Z, and copy it to storage_qq
    #     lmul!(T(-1), obs.storage_qq_1)
    #     copy!(obs.storage_qq, obs.storage_qq_1)
    #     # print("At Z'Ω^{-1}Z, should be symm, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     # Update storage_qq_1 to be Z'Ω^{-1}Z L
    #     BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1) 
    #     # print("At Z'Ω^{-1}ZL, shouldnt be symm, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     # Update HL as C'(L'Z'Ω^{-1}Z ⊗ Z'Ω^{-1}ZL)KC
    #     # print("Z'Ω^{-1}ZL, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     Ct_At_kron_A_KC!(obs.HL, obs.storage_qq_1)
    #     # print("After Ct_At_kron_A_KC, obs.HL = ", obs.HL, "\n")
    #     # Scale by coefficient -1
    #     lmul!(T(-1), obs.HL)
    #     # Update storage_qq_1 to be L'Z'Ω^{-1}ZL
    #     BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
    #     # print("At L'Z'Ω^{-1}ZL, should be symm, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     # Scale by coefficient 3
    #     lmul!(T(3), obs.storage_qq_1)
    #     # Update HL as C'(L'Z'Ω^{-1}Z ⊗ Z'Ω^{-1}ZL)KC + C'(3*L'Z'Ω^{-1}ZL ⊗ Z'Ω^{-1}Z)C
    #     # Currently, storage_qq = Z'Ω^{-1}Z
    #     Ct_A_kron_B_C!(obs.HL, obs.storage_qq_1, obs.storage_qq)
    #     # print("After Ct_A_kron_B_C, obs.HL = ", obs.HL, "\n")
    #     # Temporarily multiply by -1 to make it positive definite. 
    #     # According to the derivation, there should be no such multiplication.
    #     # lmul!(T(-1), obs.HL)
    #     if any(isnan.(obs.HL))
    #         print("Z'Ω^{-1}Z, obs.storage_qq = ", obs.storage_qq, "\n")
    #         print("3L'Z'Ω^{-1}ZL, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #         print("obs.HL = ", obs.HL, "\n")
    #     end

    #     # wrt Hσ2L
    #     # already calculated above
    end

    logl
end

function loglikelihood!(
    m::blblmmModel{T},
    needgrad::Bool = false,
    needhess::Bool = false,
    updateres::Bool = false
    ) where T <: BlasReal

    logl = zero(T)
    if needgrad
        fill!(m.∇β, 0)
        fill!(m.∇σ2, 0)
        fill!(m.∇L, 0)
    end
    if needhess
        fill!(m.Hβ, 0)
        fill!(m.Hσ2, 0)
        fill!(m.HL, 0)
        fill!(m.Hσ2, 0)
    end
    # print("m.Σ=", m.Σ, "\n")
    # print("m.σ2=", m.σ2, "\n")

    for i in eachindex(m.data)
        logl += m.w[i] * loglikelihood!(m.data[i], m.β, m.σ2, m.Σ, m.ΣL, needgrad, needhess, updateres)
        if needgrad
            BLAS.axpy!(m.w[i], m.data[i].∇β, m.∇β)
            # m.∇σ2[1] += m.w[i] * m.data[i].∇σ2[1]
            BLAS.axpy!(m.w[i], m.data[i].∇σ2, m.∇σ2)
            BLAS.axpy!(m.w[i], m.data[i].∇L, m.∇L)
        end
        if needhess
            BLAS.axpy!(m.w[i], m.data[i].Hβ, m.Hβ)
            # m.∇σ2[1] += m.w[i] * m.data[i].∇σ2[1]
            BLAS.axpy!(m.w[i], m.data[i].Hσ2, m.Hσ2)
            BLAS.axpy!(m.w[i], m.data[i].HL, m.HL)
            BLAS.axpy!(m.w[i], m.data[i].Hσ2L, m.Hσ2L)
        end
    end
    # To save cost, we didn't multiply ΣL above in the expression of ∇L
    # Here we do the multiplication
    # print("Before right mul, m.∇L = ", m.∇L, "\n")
    needgrad && BLAS.trmm!('R', 'L', 'N', 'N', T(1), m.ΣL, m.∇L)

    # print("m.∇β = ", m.∇β, "\n")
    # print("m.∇σ2 = ", m.∇σ2, "\n")
    # print("m.∇L = ", m.∇L, "\n")
    # print("logl = ", logl, "\n")

    # print("m.Hβ = ", m.Hβ, "\n")
    # print("m.Hσ2 = ", m.Hσ2, "\n")
    # print("m.HL = ", m.HL, "\n")

    logl
end


function fit!(
    m::blblmmModel;
    #solver=Ipopt.IpoptSolver(print_level=6)
    # solver=NLopt.NLoptSolver(algorithm=:LN_COBYLA, maxeval=10000)
    # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
    solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000)
    )
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    npar = p + 1 + (q * (q + 1)) >> 1
    # print("npar = ", npar, "\n")
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the qxq random effect mx has (q * (q + 1)) / 2 values,
    # the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    # then there is the error variance

    # mean effects and intercept (p + 1), random effect covariance (q * q), error variance (1)
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar) # error variance should be nonnegative, will fix later
    ub = fill( Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, m)
    # starting point
    par0 = Vector{Float64}(undef, npar)
    modelpar_to_optimpar!(par0, m)
    # print("before warmstart par0 = ", par0, "\n")
    MathProgBase.setwarmstart!(optm, par0)
    # print("after setwarmstart, par0=", par0, "\n")
    # print("after setwarmstart, MathProgBase.getsolution(optm) = ", MathProgBase.getsolution(optm), "\n")
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    # print("after setwarmstart and optimpar_to_modelpar\n")
    # optimize
    MathProgBase.optimize!(optm)
    # print("after optimize!, getsolution(optm) = ", MathProgBase.getsolution(optm), "\n")
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # refresh gradient and Hessian
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    # print("after optimize!, after optimpar_to_modelpar!, m.β = ", m.β, "\n")
    # print("after optimize!, after optimpar_to_modelpar!, m.Σ = ", m.Σ, "\n")
    # print("after optimize!, after optimpar_to_modelpar!, m.σ2 = ", m.σ2, "\n")
    loglikelihood!(m, true, true, true) 
    # !! after calculating gradient, change this to loglikelihood!(m, true, false) 
    m
end

"""
    modelpar_to_optimpar!(m, par)
Translate model parameters in `m` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par::Vector,
    m::blblmmModel
    )
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    #print("modelpar_to_optimpar m.β = ", m.β, "\n")
    copyto!(par, m.β)
    par[p+1] = log(m.σ2[1]) # take log and then exp() later to make the problem unconstrained
    # print("modelpar_to_optimpar m.β = ", m.Σ, "\n")
    
    # Since modelpar_to_optimpar is only called once, it's ok to allocate Σchol
    Σchol = cholesky(Symmetric(m.Σ), Val(false); check = false)
    # By using cholesky decomposition and optimizing L, 
    # we transform the constrained opt problem (Σ is pd) to an unconstrained problem. 
    m.ΣL .= Σchol.L
    # print("In modelpar_to_optimparm, m.ΣL = ", m.ΣL, "\n")
    offset = p + 2
    @inbounds for j in 1:q
        # print("modelpar_to_optimpar m.ΣL[j, j] = ", m.ΣL[j, j], "\n")
        par[offset] = log(m.ΣL[j, j]) # only the diagonal is constrained to be nonnegative
        offset += 1
        @inbounds for i in j+1:q
            par[offset] = m.ΣL[i, j]
            offset += 1
        end
    end
    par
    # print("modelpar_to_optimpar par = ", par, "\n")
end

"""
    optimpar_to_modelpar!(m, par)
Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
    m::blblmmModel, 
    par::Vector)
    # print("Called optimpar_to_modelpar \n")
    # print("At the beginning of optimpar_to_modelpar, m.Σ = ", m.Σ, "\n")
    # print("At the beginning of optimpar_to_modelpar, par = ", par, "\n")
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    # print("p = ", p, ", q = ", q, "\n")
    copyto!(m.β, 1, par, 1, p)
    #print("optimpar_to_modelpar par = ", par, "\n")
    # copyto!(dest, do, src, so, N)
    # Copy N elements from collection src starting at offset so, 
    # to array dest starting at offset do. Return dest.
    m.σ2[1] = exp(par[p+1])
    fill!(m.ΣL, 0)
    offset = p + 2
    @inbounds for j in 1:q
        m.ΣL[j, j] = exp(par[offset])
        offset += 1
        @inbounds for i in j+1:q
            m.ΣL[i, j] = par[offset]
            offset += 1
        end
    end
    # print("optimpar_to_modelpar m.ΣL = ", m.ΣL, "\n")
    mul!(m.Σ, m.ΣL, transpose(m.ΣL))
    # print("optimpar_to_modelpar, After translating optimpar to modelpar, m.Σ = ", m.Σ, "\n")
    # updates Σchol so that when we call loglikelihood!(), we are passing the updated cholesky
    # m.Σchol = cholesky(Symmetric(m.Σ), Val(true); check = false)
    # Σchol = cholesky(Symmetric(m.Σ), Val(true); check = false)
    # print("optimpar_to_modelpar m.Σ = ", m.Σ, "\n")
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
    # print("in eval_f, par = ", par, "\n")
    optimpar_to_modelpar!(m, par)
    # print("Inside eval_f \n")
    # print("m.β = ", m.β, "\n")
    # print("m.σ2[1] = ", m.σ2[1], "\n")
    # print("m.Σ = ", m.Σ, "\n")
    loglikelihood!(m, false, false, true)
end


function MathProgBase.eval_grad_f(
    m::blblmmModel, 
    grad::Vector, 
    par::Vector)
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    optimpar_to_modelpar!(m, par)
    loglikelihood!(m, true, false, false)
    # gradient wrt β
    copyto!(grad, m.∇β)
    # gradient wrt log(σ2)
    grad[p+1] = m.∇σ2[1] * m.σ2[1]
    offset = p + 2
    # gradient wrt log(diag(L)) and off-diag(L)
    @inbounds for j in 1:q
        # On the diagonal, gradient wrt log(ΣL[j,j])
        grad[offset] = m.∇L[j, j] * m.ΣL[j, j]
        offset += 1
        @inbounds for i in j+1:q
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
    # Get the linear indices of the upper-triangular of the non-zero blocks
    npar = ◺(m.p) + 1 + ◺(m.q) + ◺(◺(m.q))
    #       ββ      σ2σ2  σ2L       LL
    arr1 = Vector{Int}(undef, npar)
    arr2 = Vector{Int}(undef, npar)
    idx = 1
    # Hβ
    for j in 1:m.p
        for i in 1:j
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # Hσ2
    arr1[idx] = m.p + 1
    arr2[idx] = m.p + 1
    idx += 1
    # HL
    for j in (m.p+2):(m.p + 1 + ◺(m.q))
        for i in (m.p+2):j
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # Hσ2L
    for j in (m.p+2):(m.p + 1 + ◺(m.q))
        arr1[idx] = m.p + 1 # same row idx as σ2
        arr2[idx] = j
        idx += 1
    end
    return (arr1, arr2)
end

"""
    ◺(n::Integer)
Get the indices of the diagonal elements of a lower triangular matrix.
"""
function diag_idx(n::Integer)
    idx = zeros(n)
    idx[1] = 1
    for i in 2:n
        idx[i] = idx[i-1] + (n - (i-2))
    end
    return idx
end

function MathProgBase.eval_hesslag(
    m::blblmmModel, 
    H::Vector{T},
    par::Vector{T}, 
    σ::T, 
    μ::Vector{T}) where {T}    
    # l, q◺ = m.l, ◺(m.q)
    optimpar_to_modelpar!(m, par)
    loglikelihood!(m, true, true, false)
    idx = 1
    @inbounds for j in 1:m.p, i in 1:j
        H[idx] = m.Hβ[i, j]
        idx += 1
    end
    # hessian wrt log(σ2)
    # old code: H[idx] = m.Hσ2[1]
    H[idx] = m.Hσ2[1] * m.σ2[1]^2
    idx += 1
    
    # Hessian wrt diagonal: log(ΣL[j,j]), and off-diagonal: ΣL[i,j]
    # diagidx = diag_idx(m.q)
    # for (di, j) in enumerate(diagidx)
    #     m.HL[di, :] = m.HL[di, :] * m.ΣL[j, j]
    #     m.HL[:, di] = m.HL[:, di] * m.ΣL[j, j]
    #     m.HL[di, di] += m.∇L[j, j]
    # end
    m.HL[1, :] = m.HL[1, :] * m.ΣL[1, 1]
    m.HL[:, 1] = m.HL[:, 1] * m.ΣL[1, 1]
    # m.HL[1, 1] += m.∇L[1, 1] * m.ΣL[1, 1]
    m.HL[3, :] = m.HL[3, :] * m.ΣL[2, 2]
    m.HL[:, 3] = m.HL[:, 3] * m.ΣL[2, 2]
    # m.HL[3, 3] += m.∇L[2, 2] * m.ΣL[2, 2]
    @inbounds for j in 1:◺(m.q)
        # On the diagonal we have hessian wrt log(ΣL[j,j])
        H[idx] = m.HL[j, j]
        idx += 1
        for i in (j+1):◺(m.q)
            H[idx] = m.HL[i, j] 
            idx += 1
        end
    end
    # Hessian cross term
    @inbounds for j in 1:m.q
        # On the diagonal, wrt log(σ2) and log(ΣL[j,j])
        H[idx] = m.Hσ2L[j, j] * m.σ2[1] * m.ΣL[j, j]
        idx += 1
        # Off-diagonal, wrt log(σ2) and ΣL[i,j]
        for i in (j+1):m.q
            H[idx] = m.Hσ2L[i, j] * m.σ2[1]
            idx += 1
        end
    end
    lmul!(σ, H)
end