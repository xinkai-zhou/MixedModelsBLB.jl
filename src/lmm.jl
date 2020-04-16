


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
    # ΣL::LowerTriangular{T},
    needgrad::Bool = false
    ) where T <: BlasReal

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    if needgrad
        fill!(obs.∇β, T(0))
        fill!(obs.∇τ, T(0))
        fill!(obs.∇L, T(0))
    end
    ###########
    # objective
    ###########
    update_res!(obs, β)
    # Here we compute the inverse of Σ. To avoid allocation, first copy Σ to storage_qq,
    # then calculate the in-place cholesky, then in-place inverse (only the upper-tri in touched)
    copyto!(obs.storage_qq, Σ)
    # print("before potrf, obs.storage_qq = ", obs.storage_qq, "\n")
    LAPACK.potrf!('U', obs.storage_qq) # in-place cholesky
    # Calculate the logdet(Σ) part of logl
    logl = 0
    # print("after potrf, obs.storage_qq = ", obs.storage_qq, "\n")
    @inbounds for i in 1:q
        if obs.storage_qq[i, i] <= 0
            logl = -Inf
            return logl
        else 
            # (-1//2) logdet(Σ) = (-1//2) \Sum 2*log(obs.storage_qq[i, i])
            #                   = - \Sum log(obs.storage_qq[i, i])
            logl -= log(obs.storage_qq[i, i])    
        end
    end
    LAPACK.potri!('U', obs.storage_qq) # in-place inverse (only the upper-tri in touched.)
    # obs.storage_qq = Σ^{-1} + τZ'Z
    BLAS.axpy!(τ[1], obs.ztz, obs.storage_qq)
    # BLAS.syrk!('U', 'T',  τ[1], obs.Z, T(1), obs.storage_qq) # only the upper-tri is touched
    LinearAlgebra.copytri!(obs.storage_qq, 'U')
    storage_qq_chol = cholesky!(obs.storage_qq, Val(true); check = false) 
    if rank(storage_qq_chol) < q # Since storage_qq_chol is of Cholesky type, rank doesn't call SVD
        logl = -Inf # set logl to -Inf and return
        return logl
    end
    # (Σ^{-1} + τZ'Z)^{-1} Z'
    ldiv!(obs.storage_qn, storage_qq_chol, transpose(obs.Z))
    # -τ^2 * Z (Σ^{-1} + τZ'Z)^{-1} Z'
    BLAS.gemm!('N', 'N', -τ[1]^2, obs.Z, obs.storage_qn, T(0), obs.storage_nn)
    @inbounds for i in 1:n
        obs.storage_nn[i, i] += τ[1]
    end
    # Now obs.storage_nn is Ω^{-1}.
    BLAS.gemv!('T', T(1), obs.storage_nn, obs.res, 0., obs.storage_n1)
    # Currently logl equals logdet(Σ)
    logl += (-1//2) * (logdet(storage_qq_chol) + n * log(1/τ[1]) + dot(obs.res, obs.storage_n1))
    # print("New way logl = ", logl, "\n")

    ###########
    # gradient
    ###########
    if needgrad
        # wrt β
        # copyto!(obs.∇β, vec(BLAS.gemm('T', 'N', obs.X, obs.storage_n1)))
        BLAS.gemv!('T', T(1), obs.X, obs.storage_n1, T(0), obs.∇β)
        # wrt L, New code using Woodbury
        # \Omegabf^{-1}\Zbf_i
        BLAS.gemm!('N', 'N', T(1), obs.storage_nn, obs.Z, T(0), obs.storage_nq)
        #  -\Zbf_i' \Omegabf^{-1}\Zbf_i 
        BLAS.gemm!('T', 'N', T(-1), obs.Z, obs.storage_nq, T(0), obs.∇L)
        # \Zbf_i'\Omegabf^{-1}\rbf 
        BLAS.gemv!('T', T(1), obs.storage_nq, obs.res, T(0), obs.storage_1q)
        # -\Zbf_i' \Omegabf^{-1}\Zbf_i + \Zbf_i'\Omegabf^{-1}\rbf \rbf'\Omegabf^{-1}\Zbf_i
        BLAS.ger!(1., obs.storage_1q, obs.storage_1q, obs.∇L)
        # new code with Woodbury
        obs.∇τ[1] = (1/(2 * τ[1]^2)) * (tr(obs.storage_nn) - dot(obs.storage_n1, obs.storage_n1))
    end
    # print("new way ∇τ = ", obs.∇τ[1], "\n")
    # print("new way ∇L", obs.∇L, "\n")

    # ############################################################################
    # # Without Woodbury
    # ############################################################################
    # # old way of calculating the objective 
    # update_res!(obs, β)
    # mul!(obs.storage_qn, Σ, transpose(obs.Z))
    # mul!(obs.V, obs.Z, obs.storage_qn)
    # # V = obs.Z * Σ * obs.Z' + (1 / τ) * I
    # # calculate once 
    # τ_inv = (1 / τ[1])
    # @inbounds for i in 1:n
    #     obs.V[i, i] += τ_inv 
    # end
    # # Using the cholesky appraoch
    # Vchol = cholesky!(Symmetric(obs.V), Val(true); check = false) 
    # if rank(Vchol) < n # Since Vchol is of Cholesky type, rank(Vchol) doesn't call SVD
    #     logl = -Inf # set logl to -Inf and return
    #     return logl
    # end
    # ldiv!(obs.storage_n1, Vchol, obs.res)
    # logl = (-1//2) * (logdet(Vchol) + dot(obs.res, obs.storage_n1))
    # print("Old way logl = ", logl, "\n")

    # # gradient
    # if needgrad
    #     # wrt β
    #     # copyto!(obs.∇β, vec(BLAS.gemm('T', 'N', obs.X, obs.storage_n1)))
    #     BLAS.gemv!('T', T(1), obs.X, obs.storage_n1, T(0), obs.∇β)
    #     # old code for gradient wrt L
    #     ldiv!(obs.storage_nq, Vchol, obs.Z)
    #     BLAS.gemm!('T', 'N', -1., obs.Z, obs.storage_nq, 0., obs.∇L)
    #     BLAS.gemv!('T', 1., obs.storage_nq, obs.res, 0., obs.storage_1q)
    #     BLAS.ger!(1., obs.storage_1q, obs.storage_1q, obs.∇L)
    #     # # old code for gradient wrt τ
    #     # # Since Vchol and V are no longer needed, we can calculate in-place inverse of obs.V
    #     # # obs.V holds the in place cholesky of V
    #     LAPACK.potri!('U', obs.V)
    #     obs.∇τ[1] = (1/(2 * τ[1]^2)) * (tr(obs.V) - dot(obs.storage_n1, obs.storage_n1))
    #     # # ldiv!(obs.storage_nn, Vchol, obs.I_n)
    #     # # obs.∇τ[1] = (1/(2 * τ[1]^2)) * (tr(obs.storage_nn) - dot(obs.storage_n1, obs.storage_n1))
    # end
    # print("old way ∇τ = ", obs.∇τ[1], "\n")
    # print("old way ∇L", obs.∇L, "\n")

    # Why can't we use autodiff???
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