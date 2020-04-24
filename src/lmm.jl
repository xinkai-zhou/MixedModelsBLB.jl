


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
    τ::Vector{T}, # inverse of linear regression variance
    Σ::Matrix{T},
    ΣL::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    τinv = 1 / τ[1]
    if needgrad
        fill!(obs.∇β, T(0))
        fill!(obs.∇τ, T(0))
        fill!(obs.∇L, T(0))
    end
    ###########
    # objective
    ###########
    update_res!(obs, β)
    copy!(obs.storage_qq, obs.ztz) 
    # L'Z'Z
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq)
    needgrad && copy!(obs.∇L, obs.storage_qq)
    # τ L'Z'Z L
    BLAS.trmm!('R', 'L', 'N', 'N', τ[1], ΣL, obs.storage_qq)
    # storage_qq_1 = τ L'Z'Z L
    needgrad && copy!(obs.storage_qq_1, obs.storage_qq) 
    # I + L'Z'ZL
    @inbounds for i in 1:q
        obs.storage_qq[i, i] += T(1)
    end
    LAPACK.potrf!('U', obs.storage_qq) # cholesky on I + L'Z'ZL
    # Z'r
    BLAS.gemv!('T', T(1), obs.Z, obs.res, T(0), obs.storage_q)
    # L'Z'r
    BLAS.trmv!('L', 'T', 'N', ΣL, obs.storage_q)
    # chol^{-1} L'Z'r
    # BLAS.trsv!(ul, tA, dA, obs.storage_qq, b)
    BLAS.trsv!('U', 'T', 'N', obs.storage_qq, obs.storage_q)
    logl = 0
    @inbounds for i in 1:q
        if obs.storage_qq[i, i] <= 0
            logl = -Inf
            return logl
        else 
            # Calculate logdet(I + L'Z'ZL) through cholesky.
            logl -= log(obs.storage_qq[i, i])    
        end
    end
    logl += (-1//2) * (n * log(1/τ[1] + τ[1] * dot(obs.res, obs.res) - 
                        τ[1]^2 * dot(obs.storage_q, obs.storage_q)))
    
    
    
    ###########
    # gradient
    ###########
    if needgrad
        # currently, storage_q = chol^{-1} L'Z'r
        # update storage_q to (I+τLZ'ZL)^{-1} L'Z'r
        BLAS.trsv!('U', 'N', 'N', obs.storage_qq, obs.storage_q)
        # update storage_q to L(I+τLZ'ZL)^{-1} L'Z'r
        BLAS.trmv!('L', 'N', 'N', ΣL, obs.storage_q)

        # wrt β
        # First calculate τX'r
        BLAS.gemv!('T', τ[1], obs.X, obs.res, T(0), obs.∇β)
        # then, obs.∇β = τX'r + X'Z L(I+τLZ'ZL)^{-1} L'Z'r
        BLAS.gemv!('N', -τ[1]^2, obs.xtz, obs.storage_q, T(1), obs.∇β)

        # wrt τ
        # Since we no longer need obs.res, update obs.res to be Ω^{-1} r
        BLAS.gemv!('N', -τ[1]^2, obs.Z, obs.storage_q, τ[1], obs.res)
        # To evaluate tr(Ω^{-1}), Calculate (I+τLZ'ZL)^{-1} L'Z'ZL 
        # Currently, storage_qq_1 = τ L'Z'Z L
        BLAS.trsm!('L', 'U', 'T', 'N', 1/τ[1], obs.storage_qq, obs.storage_qq_1)
        BLAS.trsm!('L', 'U', 'N', 'N', T(1), obs.storage_qq, obs.storage_qq_1)
        needhess && obs.Hτ[1, 1] = (τinv^2 / 2) * n - τinv^2 * tr(obs.storage_qq_1)
        obs.∇τ[1] = (τinv / 2) * n - (1//2) * tr(obs.storage_qq_1) - 
                    (τinv^2 / 2) * dot(obs.res, obs.res)


        # Currently, storage_qq is holding the chol factor
        # Before it is destroyed, we need to calculate chol^{-1} L'Z'X for the hessian of β
        if needhess
            copy!(obs.storage_qp, obs.ztx)
            # L'Z'X
            BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qp)
            # chol^{-1} L'Z'X 
            BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.storage_qp)
        end

        # wrt L
        # Currently, obs.∇L = L'Z'Z
        # Calculate chol^{-1}L'Z'Z and store it in obs.∇L
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.∇L)
        ## Before destroying storage_qq_1, we need to calculate its product with 
        ## itself for the hessian of \tau. Since storage_qq is no longer
        ## needed, we overwrite it with a rank-k update
        needhess && BLAS.gemm!('N', 'N', T(1), obs.storage_qq_1, obs.storage_qq_1, T(0), obs.storage_qq)
        needhess && obs.Hτ[1, 1] += (1//2) * tr(obs.storage_qq)
        # Since we no longer need storage_qq_1, we overwrite it with a rank-k update:
        # storage_qq_1 = τ^2 * Z'ZL(I+τLZ'ZL)^{-1}L'Z'Z
        BLAS.syrk!('U', 'T', τ[1]^2, obs.∇L, T(0), obs.storage_qq_1)
        # Since syrk only updated the upper triangle, do copytri
        LinearAlgebra.copytri!(obs.storage_qq_1, 'U') 
        # Update obs.storage_qq_1 as -Z'Ω^{-1}Z = -τZ'Z + τ^2 * Z'ZL(I+τLZ'ZL)^{-1}L'Z'Z
        BLAS.axpy!(-τ[1], obs.ztz, obs.storage_qq_1)
        # copy it to obs.∇L
        copy!(obs.∇L, obs.storage_qq_1)
        # Currently res = Ω^{-1} r
        # Update storage_q as Z'Ω^{-1} r
        BLAS.gemv!('T', T(1), obs.Z, obs.res, T(0), obs.storage_q)
        BLAS.syr!('U', T(1), obs.storage_q, obs.∇L)
        LinearAlgebra.copytri!(obs.∇L, 'U')
        # In fact, this is just the gradient wrt Σ
        # The right multiplication of L will be done once in the aggregation step
        # to save computation. 
    end

    ###########
    # hessian
    ###########
    if needhess
        # wrt β
        # Currently, storage_qp = chol^{-1} L'Z'X.
        copy!(obs.Hβ, obs.xtx)
        # Update Hβ as -τX'X + τ^2 * X'ZL(I+τLZ'ZL)^{-1}L'Z'X through rank-k update
        BLAS.syrk!('U', 'T', τ[1]^2, obs.storage_qp, -τ[1], obs.Hβ)
        
        # wrt τ
        # already calculated above

        # wrt vech L 
        # Currently storage_qq_1 = -Z'Ω^{-1}Z
        copy!(obs.storage_qq, obs.storage_qq_1)
        # Update storage_qq_1 to be L'Z'Ω^{-1}Z
        BLAS.trmm!('L', 'L', 'T', 'N', T(-1), ΣL, obs.storage_qq_1)
        # Update HL as C'(L'Z'Ω^{-1}Z ⊗ Z'Ω^{-1}ZL)KC
        Ct_At_kron_A_KC!(obs.HL, obs.storage_qq_1)
        # Scale by coefficient -1
        lmul!(T(-1), obs.HL)
        # Update storage_qq_1 to be L'Z'Ω^{-1}ZL
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
        # Scale by coefficient 3
        lmul!(T(3), obs.storage_qq_1)
        # Update HL as C'(L'Z'Ω^{-1}Z ⊗ Z'Ω^{-1}ZL)KC + C'(3*L'Z'Ω^{-1}ZL ⊗ Z'Ω^{-1}Z)C
        Ct_A_kron_B_C!(obs.HL, obs.storage_qq_1, obs.storage_qq)
    end

    logl
end

function loglikelihood!(
    m::blblmmModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    logl = zero(T)
    if needgrad
        fill!(m.∇β, 0)
        fill!(m.∇τ, 0)
        fill!(m.∇L, 0)
    end
    # print("m.Σ=", m.Σ, "\n")
    # print("m.τ=", m.τ, "\n")
    if needgrad
        @inbounds for i = 1:length(m.data)
            logl += m.w[i] * loglikelihood!(m.data[i], m.β, m.τ, m.Σ, needgrad)
            # m.∇β .+= m.w[i] .* m.data[i].∇β
            BLAS.axpy!(m.w[i], m.data[i].∇β, m.∇β)
            m.∇τ[1] += m.w[i] * m.data[i].∇τ[1]
            # m.∇L .+= m.w[i] .* m.data[i].∇L
            BLAS.axpy!(m.w[i], m.data[i].∇L, m.∇L)
        end
        # Here we multiply ΣL once.
        rmul!(m.∇L, m.ΣL)
        # BLAS.trmm!('R', 'L', 'N', 'N', T(1), m.ΣL, m.∇L)
        # 'R', 'L', 'N', 'N': right side, lower, no transpose, use the diagonal of m.ΣL
    else
        @inbounds for i = 1:length(m.data)
            logl += m.w[i] * loglikelihood!(m.data[i], m.β, m.τ, m.Σ, needgrad)
        end
    end
    
    # print("In loglikelihood(m), m.∇β=", m.∇β, "\n")
    # print("In loglikelihood(m), m.∇τ[1]=", m.∇τ[1], "\n")
    # print("In loglikelihood(m), m.∇L=", m.∇L, "\n")
    # print("m.τ[1] = ", m.τ, "\n")
    # print("logl = ", logl, "\n")
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
    # print("after optimize!, after optimpar_to_modelpar!, m.τ = ", m.τ, "\n")
    loglikelihood!(m, false, false) 
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
    par[p+1] = log(m.τ[1]) # take log and then exp() later to make the problem unconstrained
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
    m.τ[1] = exp(par[p+1])
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
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(m::blblmmModel) = [:Grad]

function MathProgBase.eval_f(
    m::blblmmModel, 
    par::Vector)
    # print("in eval_f, par = ", par, "\n")
    optimpar_to_modelpar!(m, par)
    # print("Inside eval_f \n")
    # print("m.β = ", m.β, "\n")
    # print("m.τ[1] = ", m.τ[1], "\n")
    # print("m.Σ = ", m.Σ, "\n")
    loglikelihood!(m, false, false)
end


function MathProgBase.eval_grad_f(
    m::blblmmModel, 
    grad::Vector, 
    par::Vector)
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    optimpar_to_modelpar!(m, par)
    loglikelihood!(m, true, false)
    # gradient wrt β
    copyto!(grad, m.∇β)
    # gradient wrt log(τ)
    grad[p+1] = m.∇τ[1] * m.τ[1]
    # gradient wrt log(diag(L)) and off-diag(L)
    offset = p + 2
    # print("In eval_grad_f, m.ΣL = ", m.ΣL, "\n")
    @inbounds for j in 1:q
        grad[offset] = m.∇L[j, j] * m.ΣL[j, j]
        offset += 1
        @inbounds for i in j+1:q
            grad[offset] = m.∇L[i, j]
            offset += 1
        end
    end
    # print("par = ", par, "\n")
    # print("grad = ", grad, "\n")
    nothing
end