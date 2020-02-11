

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
    # print("m.XtX = ", m.XtX, "\n")
    # least square solution for β
    ldiv!(m.β, cholesky(Symmetric(m.XtX)), xty)
    # ldiv!(Y, A, B) -> Y
    # Compute A \ B in-place and store the result in Y, returning the result.

    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
        rss += abs2(norm(m.data[i].res))
    end
    m.τ[1] = m.ntotal / rss # τ is the inverse of error variance
    # we used the inverse so that the objective function is convex
    m.β
end


"""
init_MoM(m)
Initialize model parameters by the method of moments (MoM). 
For β, the MoM estimator is the same as the OLS estimator.
"""
function init_MoM!(
    m::blblmmModel{T}
    ) where T <: BlasReal

    # OLS for β
    # accumulate sufficient statistics X'y
    xty = zeros(T, m.p) 
    for i in eachindex(m.data)
        BLAS.gemv!('T', one(T), m.data[i].X, m.data[i].y, one(T), xty)
        # gemv!(tA, alpha, A, x, beta, y) 
        # Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA. 
        # alpha and beta are scalars. Return the updated y.
    end
    # print("m.XtX = ", m.XtX, "\n")
    # least square solution for β
    ldiv!(m.β, cholesky(Symmetric(m.XtX)), xty)
    # ldiv!(Y, A, B) -> Y
    # Compute A \ B in-place and store the result in Y, returning the result.

    # For τ 
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
        rss += abs2(norm(m.data[i].res))
    end
    m.τ[1] = m.ntotal / rss # τ is the inverse of error variance
    # we used the inverse so that the objective function is convex

    # MoM Σ
    m.Σ .= 0
    for i in eachindex(m.data)
        # plainly translating the expression
        m.Σ .+= 
            LinearAlgebra.inv(m.data[i].ztz) * 
            transpose(m.data[i].Z) * 
            m.data[i].res *
            transpose(m.data[i].res) *
            m.data[i].Z *
            LinearAlgebra.inv(m.data[i].ztz) .- 
            (1 / m.τ[1]) .* LinearAlgebra.inv(m.data[i].ztz)

        # # inverse of ztz
        # copyto!(m.data[i].storage_qq, LinearAlgebra.inv(m.data[i].ztz))
        # copyto!(m.Σ, - (1 / m.τ[1]) .* m.data[i].storage_qq)
        # # calculate z(ztz)^-1
        # mul!(m.data[i].storage_nq, m.data[i].Z, m.data[i].storage_qq)
        # # calculate rt * z(ztz)^-1
        # # transpose(m.data[i].res) is a vector. this may create trouble.
        # mul!(m.data[i].storage_1q, transpose(m.data[i].res), m.data[i].storage_nq)
        # # copyto!(storage_1q, BLAS.gemm('Y', 'N', m.data[i].res, m.data[i.Z]))
        # # can use gemv
        # mul!(m.data[i].storage_qq, transpose(m.data[i].storage_1q), m.data[i].storage_1q)
        # m.Σ .+= m.data[i].storage_qq
    end
    # print("init_MoM m.Σ = ", m.Σ, "\n")

    # m.β, m.Σ 
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

"""
update_w!(m, w)
Update the weight vector using w
"""
function update_w!(
    m::blblmmModel{T}, 
    w::Vector{T}
    ) where T <: BlasReal
    # m.w = w # don't do this bcz it's pointing to the memory of w
    # so if we change w, then m.w will also change
    copyto!(m.w, w)
end



function loglikelihood!(
    obs::blblmmObs{T},
    β::Vector{T},
    τ::Vector{T}, # inverse of linear regression variance
    Σ::Matrix{T},
    needgrad::Bool = false
    # needhess::Bool = false
    ) where T <: BlasReal

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    if needgrad
        fill!(obs.∇β, 0)
        fill!(obs.∇τ, 0)
        fill!(obs.∇L, 0)
    end
    # if needhess
    #     fill!(obs.Hβ, 0)
    #     fill!(obs.Hτ, 0)
    #     fill!(obs.HL, 0) 
    # end    

    # evaluate the loglikelihood
    update_res!(obs, β)
    # V = obs.Z * Σ * obs.Z' # this creates V everytime loglik is evaluated. 
    # We avoid it by adding V to blblmmObs type
    mul!(obs.storage_qn, Σ, transpose(obs.Z))
    mul!(obs.V, obs.Z, obs.storage_qn)
    # V = obs.Z * Σ * obs.Z' + (1 / τ) * I
    for i in 1:n
        obs.V[i, i] += (1 / τ[1]) # instead of τ
    end
    
    try
        Vchol = cholesky(Symmetric(obs.V))
    catch
        print("Σ=", Σ, "\n")
        print("τ[1]=", τ[1], "\n")
        print("obs.V=", obs.V, "\n")
    end
    Vchol = cholesky(Symmetric(obs.V))
    ldiv!(obs.storage_n1, Vchol, obs.res)
    logl = (-1//2) * (logdet(Vchol) + dot(obs.res, obs.storage_n1))
    # Since we will use obs.V later for gradient, we cannot do in place cholesky as below.
    # !!!!! REWRITE IT. THE FOLLOWING CODE HAS BUGS!
    # cholesky!(Symmetric(obs.V)) # in place cholesky
    # ldiv!(obs.storage_n1, obs.V, obs.res)
    # logl = (-1//2) * (logdet(obs.V) + dot(obs.res, obs.storage_n1))

    # gradient
    if needgrad
        # wrt β
        copyto!(obs.∇β, vec(BLAS.gemm('T', 'N', obs.X, obs.storage_n1)))
        # wrt τ
        # obs.∇τ[1] = (1/(2 * τ[1]^2)) * (sum(diag(inv(obs.V))) - transpose(obs.res) * inv(obs.V) * inv(obs.V) * obs.res)
        obs.∇τ[1] = (1/(2 * τ[1]^2)) * (sum(diag(inv(Vchol))) - dot(obs.storage_n1, obs.storage_n1))
        # print("obs.∇τ[1]=", obs.∇τ[1], "\n")
        # wrt L
        L = cholesky(Symmetric(Σ)).L 
        # !!!!!!!! this step is repeated for every data point.  we should avoid this! 
        ldiv!(obs.storage_nq, Vchol, obs.Z)
        # obs.storage_nq = Vchol \ obs.Z
        copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.Z, obs.storage_nq))
        rmul!(obs.storage_qq, L) # Calculate AB, overwriting A. B must be of special matrix type.
        obs.∇L .= - obs.storage_qq
        copyto!(obs.storage_1q, BLAS.gemm('T', 'N', reshape(obs.res, (n, 1)), obs.storage_nq))
        copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.storage_1q, obs.storage_1q))
        rmul!(obs.storage_qq, L) # Calculate AB, overwriting A. B must be of special matrix type.
        obs.∇L .+= obs.storage_qq 
        # print("obs.∇L=", obs.∇L, "\n")
        # if counter == 50
        #     print("obs.∇β=", obs.∇β, "\n")
        #     print("obs.∇τ[1]=", obs.∇τ[1], "\n")
        #     print("obs.∇L=", obs.∇L, "\n")
        #     print("(1/(2 * τ[1]^2)) = ", (1/(2 * τ[1]^2)), "\n")
        #     print("obs.n=", n, "\n")
        #     print("sum(diag(inv(Vchol))) = ", sum(diag(inv(Vchol))), "\n")
        #     print("dot(obs.storage_n1, obs.storage_n1) = ", dot(obs.storage_n1, obs.storage_n1), "\n")
        # end
    end
    # ??? Why can't we use autodiff???

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
    if needgrad
        fill!(m.∇β, 0)
        fill!(m.∇τ, 0)
        fill!(m.∇L, 0)
    end
    # print("m.Σ=", m.Σ, "\n")
    # print("m.τ=", m.τ, "\n")
    if needgrad
        for i = 1:length(m.data)
            logl += m.w[i] * loglikelihood!(m.data[i], m.β, m.τ, m.Σ, needgrad)#, needhess)
            m.∇β .+= m.w[i] .* m.data[i].∇β
            m.∇τ[1] += m.w[i] * m.data[i].∇τ[1]
            m.∇L .+= m.w[i] .* m.data[i].∇L
        end
    else
        for i = 1:length(m.data)
            logl += m.w[i] * loglikelihood!(m.data[i], m.β, m.τ, m.Σ, needgrad)#, needhess)
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
    print("npar = ", npar, "\n")
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

    # print("after setwarmstart, MathProgBase.getsolution(optm) = ", MathProgBase.getsolution(optm), "\n")
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
    Σchol = cholesky(Symmetric(m.Σ))
    #print("modelpar_to_optimpar Σchol = ", Σchol, "\n")
    # By using cholesky decomposition and optimizing L, 
    # we transform the constrained opt problem (Σ is pd) to an unconstrained problem. 
    m.ΣL .= Σchol.L
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
    # print("modelpar_to_optimpar par = ", par, "\n")
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
    #print("optimpar_to_modelpar par = ", par, "\n")
    # copyto!(dest, do, src, so, N)
    # Copy N elements from collection src starting at offset so, 
    # to array dest starting at offset do. Return dest.
    m.τ[1] = exp(par[p+1])
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
    for j in 1:q
        grad[offset] = m.∇L[j, j] * m.ΣL[j, j]
        offset += 1
        for i in j+1:q
            grad[offset] = m.∇L[i, j]
            offset += 1
        end
    end
    # print("par = ", par, "\n")
    # print("grad = ", grad, "\n")
    nothing
end