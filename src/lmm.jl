

"""
init_β(m)
Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
squares solution.
"""
function init_β!(
    m::blblmmModel{T}
    ) where T <: BlasReal
    # accumulate sufficient statistics X'y
    xty = zeros(T, m.p) 
    for i in eachindex(m.data)
        BLAS.gemv!('T', one(T), m.data[i].X, m.data[i].y, one(T), xty)
        # gemv!(tA, alpha, A, x, beta, y) 
        # Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA. 
        # alpha and beta are scalars. Return the updated y.
    end
    # least square solution for β
    ldiv!(m.β, cholesky(Symmetric(m.XtX)), xty)
    # ldiv!(Y, A, B) -> Y
    # Compute A \ B in-place and store the result in Y, returning the result.
    # The argument A should not be a matrix. 
    # Rather, instead of matrices it should be a factorization object (e.g. produced by factorize or cholesky). 
    # The reason for this is that factorization itself is both expensive and typically allocates memory 
    # (although it can also be done in-place via, e.g., lu!), 
    # and performance-critical situations requiring ldiv! usually also require fine-grained control 
    # over the factorization of A.

    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
        rss += abs2(norm(m.data[i].res))
    end
    m.τ[1] = m.ntotal / rss # τ[1] is the inverse of error variance
    m.β
end

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
    for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
    end
    nothing
end
# function standardize_res!(
#     obs::blblmmObs{T}, 
#     σinv::T
#     ) where T <: BlasReal
#     obs.res .*= σinv
# end
# function standardize_res!(
#     m::blblmmModel{T}
#     ) where T <: BlasReal
#     σinv = sqrt(m.τ[1])
#     # standardize residual
#     for i in eachindex(m.data)
#         standardize_res!(m.data[i], σinv)
#     end
#     nothing
# end

"""
update_w!(m, w)
Update the weight vector using w
"""
function update_w!(
    m::blblmmModel{T}, w
) where T <: BlasReal
    # m.w = w # don't do this bcz it's pointing to the memory of w
    # so if we change w, then m.w will also change
    copyto!(m.w, w)
    nothing
end



function loglikelihood!(
    obs::blblmmObs{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    Σ::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    if needgrad
        fill!(obs.∇β, 0)
        fill!(obs.∇τ, 0)
        fill!(obs.∇Σ, 0)
    end
    if needhess
        fill!(obs.Hβ, 0)
        fill!(obs.Hτ, 0)
        fill!(obs.HΣ, 0)
    end    

    # evaluate the loglikelihood
    update_res!(obs, β)
    # V = obs.Z * Σ * obs.Z' # this creates V everytime loglik is evaluated. 
    # We avoid it by adding V to blblmmObs type
    mul!(obs.storage_qn, Σ, transpose(obs.Z))
    mul!(obs.V, obs.Z, obs.storage_qn)
    # V = obs.Z * Σ * obs.Z' + (1 / τ[1]) * I
    for i in 1:n
        obs.V[i, i] += (1 / τ[1])
    end
    # ?? put Vchol in the blblmmObs type
    Vchol = cholesky(Symmetric(obs.V))
    print(Vchol)
    logl = (-1//2) * (logdet(Vchol) + dot(obs.res, Vchol \ obs.res))

    # gradient
    # if needgrad
    #     # wrt β
    #     mul!(obs.∇β, transpose(obs.X), obs.res)
    #     BLAS.gemv!('N', -inv(1 + qf), obs.xtz, obs.storage_q2, one(T), obs.∇β)
    #     obs.∇β .*= sqrtτ
    #     # wrt τ
    #     obs.∇τ[1] = (n - rss + 2qf / (1 + qf)) / 2τ
    #     # wrt Σ
    #     copyto!(obs.∇Σ, obs.ztz)
    #     BLAS.syrk!('U', 'N', (1//2)inv(1 + qf), obs.storage_q1, (-1//2)inv(1 + tr), obs.∇Σ)
    #     copytri!(obs.∇Σ, 'U')
    # end
    # # Hessian: TODO
    # if needhess; end;

    # output
    logl
end

function loglikelihood!(
    m::blblmmModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    logl = 0
    for i = 1:length(m.data)
        logl += m.w[i] * loglikelihood!(m.data[i], m.β, m.τ[1], m.Σ, needgrad, needhess)
    end
end


function fit!(
    m::blblmmModel,
    solver=Ipopt.IpoptSolver(print_level=0)
    )
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    npar = p + 1 + (q * (q + 1)) >> 1
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
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # refresh gradient and Hessian
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
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
    copyto!(par, m.β)
    par[p+1] = log(m.τ[1]) # take log and then exp() later to make the problem unconstrained
    Σchol = cholesky(Symmetric(m.Σ))
    m.ΣL .= Σchol.L
    # ?? what is the benefit of cholesky here?
    offset = p + 2
    for j in 1:q
        par[offset] = log(m.ΣL[j, j]) # only the diagonal is constrained to be nonnegative
        offset += 1
        for i in j+1:q
            par[offset] = m.ΣL[i, j]
            offset += 1
        end
    end
    par
end

"""
    optimpar_to_modelpar!(m, par)
Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
    m::blblmmModel, 
    par::Vector)
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    copyto!(m.β, 1, par, 1, p)
    # copyto!(dest, do, src, so, N)
    # Copy N elements from collection src starting at offset so, to array dest starting at offset do. Return dest.
    m.τ[1] = exp(par[p+1]) # why taking exponential?
    fill!(m.ΣL, 0)
    offset = p + 2
    for j in 1:q
        m.ΣL[j, j] = exp(par[offset])
        offset += 1
        for i in j+1:q
            m.ΣL[i, j] = par[offset]
            offset += 1
        end
    end
    mul!(m.Σ, m.ΣL, transpose(m.ΣL))
    nothing
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
    optimpar_to_modelpar!(m, par)
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
    # gradient wrt L
    mul!(m.storage_qq, m.∇Σ, m.ΣL)
    offset = p + 2
    for j in 1:q
        grad[offset] = 2m.storage_qq[j, j] * m.ΣL[j, j]
        offset += 1
        for i in j+1:q
            grad[offset] = 2(m.storage_qq[i, j] + m.storage_qq[j, i])
            offset += 1
        end
    end
    nothing
end