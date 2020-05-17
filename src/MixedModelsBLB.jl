__precompile__()

module MixedModelsBLB

using MathProgBase
using Reexport
using Distributions
using JuliaDB
using LinearAlgebra
using MixedModels
using Random
using StatsModels
using StatsBase
using DataFrames
using CSV
using InteractiveUtils
using Permutations
using Ipopt

using LinearAlgebra: BlasReal, copytri!
import LinearAlgebra: BlasFloat, checksquare

@reexport using Ipopt
@reexport using NLopt
@reexport using MixedModels

export blblmmObs, blblmmModel
export fit!, fitted, loglikelihood!, update_res!, update_w!, extract_Σ!
export blb_one_subset, blb_full_data

"""
blblmmObs
blblmmObs(y, X, Z)
A realization of BLB linear mixed model data instance.
"""
struct blblmmObs{T <: LinearAlgebra.BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T} # X should include a column of 1's
    Z::Matrix{T}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇σ2::Vector{T}   # gradient wrt σ2
    ∇L::Matrix{T}   # gradient wrt L 
    Hβ::Matrix{T}   # Hessian wrt β
    Hσ2::Vector{T}   # Hessian wrt σ2
    Hσ2L::Vector{T}   # Hessian cross term
    HL::Matrix{T}   # Hessian wrt L
    res::Vector{T}  # residual vector
    xtx::Matrix{T}  # Xi'Xi (p-by-p)
    ztz::Matrix{T}  # Zi'Zi (q-by-q)
    ztx::Matrix{T}  # Zi'Xi (q-by-p)
    storage_n::Vector{T}
    storage_q::Vector{T}
    storage_qn::Matrix{T}
    storage_nq::Matrix{T}
    storage_qq::Matrix{T}
    storage_qq_1::Matrix{T}
    storage_qp::Matrix{T}
    # storage_nn::Matrix{T}
    # V::Matrix{T}
    #Vchol::CholeskyPivoted{T}
end
#storage_q1::Vector{T}
#storage_q2::Vector{T}
# Vchol::Matrix{T}

function blblmmObs(
    y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T}
    ) where T <: BlasReal
    n, p, q = size(X, 1), size(X, 2), size(Z, 2)
    q◺ = ◺(q)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇σ2  = Vector{T}(undef, 1)
    ∇L  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hσ2  = Vector{T}(undef, 1)
    Hσ2L  = Vector{T}(undef, q◺)
    HL  = Matrix{T}(undef, q◺, q◺)
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    ztz = transpose(Z) * Z
    ztx = transpose(Z) * X
    storage_n = Vector{T}(undef, n)
    storage_q = Vector{T}(undef, q)
    storage_qn = Matrix{T}(undef, q, n)
    storage_nq = Matrix{T}(undef, n, q)
    storage_qq = Matrix{T}(undef, q, q) 
    storage_qq_1 = Matrix{T}(undef, q, q) 
    storage_qp = Matrix{T}(undef, q, p)
    # storage_nn = Matrix{T}(undef, n, n)
    # V = Matrix{T}(undef, n, n)
    # Vchol = cholesky(V, Val(true); check = false)
    blblmmObs{T}(
        y, X, Z, 
        ∇β, ∇σ2, ∇L, 
        Hβ, Hσ2, Hσ2L, HL,
        res, xtx, ztz, ztx,
        storage_n, storage_q, 
        storage_qn, storage_nq, 
        storage_qq, storage_qq_1, 
        storage_qp
    )#, storage_nn, V)
end
# constructor
#storage_q1 = Vector{T}(undef, q)
#storage_q2 = Vector{T}(undef, q)
#storage_q1, storage_q2, 

"""
blblmmModel
blblmmModel
BLB linear mixed model, which contains a vector of 
`blblmmObs` as data, model parameters, and working arrays.
"""
struct blblmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{blblmmObs{T}}
    w::Vector{T}    # a vector of weights from bootstraping the subset
    # ntotal::Int     # total number of clusters
    p::Int          # number of mean parameters in linear regression
    q::Int          # number of random effects
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    σ2::Vector{T}    # inverse of linear regression variance parameter 
    # we used the inverse so that the objective function is convex
    Σ::Matrix{T}    # q-by-q (psd) matrix
    # working arrays
    # ΣL::LowerTriangular{T, Matrix{T}}
    ΣL::Matrix{T}
    ∇β::Vector{T}   # gradient from all observations
    ∇σ2::Vector{T}
    ∇L::Matrix{T}
    Hβ::Matrix{T}   # Hessian from all observations
    Hσ2::Vector{T}
    HL::Matrix{T}
    Hσ2L::Vector{T}
    # XtX::Matrix{T}      # X'X = sum_i Xi'Xi
    # storage_qq::Matrix{T}
    # storage_nq::Matrix{T}
end

function blblmmModel(obsvec::Vector{blblmmObs{T}}) where T <: BlasReal
    n, p, q = length(obsvec), size(obsvec[1].X, 2), size(obsvec[1].Z, 2)
    q◺ = ◺(q)
    npar = p + 1 + (q * (q + 1)) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the qxq random effect mx has (q * (q + 1))/2 values
    # the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    # then there is the error variance
    w   = ones(T, n) # initialize weights to be 1
    β   = Vector{T}(undef, p)
    σ2   = Vector{T}(undef, 1)
    Σ   = Matrix{T}(undef, q, q)
    # ΣL  = LowerTriangular(Matrix{T}(undef, q, q))
    ΣL  = Matrix{T}(undef, q, q)
    ∇β  = Vector{T}(undef, p)
    ∇σ2  = Vector{T}(undef, 1)
    ∇L  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hσ2  = Vector{T}(undef, 1)
    HL  = Matrix{T}(undef, q◺, q◺)
    Hσ2L  = Vector{T}(undef, q◺)
    # XtX = zeros(T, p, p) # sum_i xi'xi
    # ntotal = 0
    # for i in eachindex(obsvec)
    #     ntotal  += length(obsvec[i].y)
    #     XtX    .+= obsvec[i].xtx
    # end
    # storage_qq = Matrix{T}(undef, q, q)
    # storage_nq = Matrix{T}(undef, n, q)
    
    blblmmModel{T}(
        obsvec, w, p, q, 
        β, σ2, Σ, ΣL, 
        ∇β, ∇σ2, ∇L, Hβ, Hσ2, HL, Hσ2L
    ) 
        # XtX, storage_qq, storage_nq)
end


include("lmm.jl")
include("blb.jl")
include("multivariate_calculus.jl")

end # module
