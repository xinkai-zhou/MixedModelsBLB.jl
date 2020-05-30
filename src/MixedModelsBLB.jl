__precompile__()

module MixedModelsBLB

using MathProgBase
using Reexport
using Distributions
using JuliaDB
using Random
using StatsModels
using StatsBase
using DataFrames
using CSV
using InteractiveUtils
using Permutations
using LinearAlgebra

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
    # grad and hess
    ∇β::Vector{T}   # gradient wrt β
    ∇σ²::Vector{T}   # gradient wrt σ²
    ∇L::Matrix{T}   # gradient wrt L 
    Hββ::Matrix{T}   # Hessian wrt β
    Hσ²σ²::Vector{T}   # Hessian wrt σ²
    Hσ²L::Vector{T}   # Hessian cross term
    HLL::Matrix{T}   # Hessian wrt L
    # pre-compute
    yty::T
    xty::Vector{T} 
    zty::Vector{T}
    xtx::Matrix{T}  # Xi'Xi (p-by-p)
    ztx::Matrix{T}  # Zi'Xi (q-by-p)
    ztz::Matrix{T}  # Zi'Zi (q-by-q)
    # working arrays
    xtr::Vector{T}
    ztr::Vector{T}
    storage_p::Vector{T}
    storage_q_1::Vector{T}
    storage_q_2::Vector{T}
    storage_qq_1::Matrix{T}
    storage_qq_2::Matrix{T}
    storage_qq_3::Matrix{T}
    storage_qp::Matrix{T}
end


function blblmmObs(
    y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T}
    ) where T <: BlasReal
    n, p, q = size(X, 1), size(X, 2), size(Z, 2)
    q◺ = ◺(q)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    ∇β     = Vector{T}(undef, p)
    ∇σ²    = Vector{T}(undef, 1)
    ∇L     = Matrix{T}(undef, q, q)
    Hββ    = Matrix{T}(undef, p, p)
    Hσ²σ²  = Vector{T}(undef, 1)
    Hσ²L   = Vector{T}(undef, q◺)
    HLL    = Matrix{T}(undef, q◺, q◺)
    yty = dot(y, y)
    xty = transpose(X) * y
    zty = transpose(Z) * y
    xtx = transpose(X) * X
    ztx = transpose(Z) * X
    ztz = transpose(Z) * Z
    xtr = Vector{T}(undef, p)
    ztr = Vector{T}(undef, q)
    storage_p = Vector{T}(undef, p)
    storage_q_1 = Vector{T}(undef, q)
    storage_q_2 = Vector{T}(undef, q)
    storage_qq_1 = Matrix{T}(undef, q, q) 
    storage_qq_2 = Matrix{T}(undef, q, q) 
    storage_qq_3 = Matrix{T}(undef, q, q) 
    storage_qp = Matrix{T}(undef, q, p)
    blblmmObs{T}(
        y, X, Z, 
        ∇β, ∇σ², ∇L, 
        Hββ, Hσ²σ², Hσ²L, HLL,
        yty, xty, zty, xtx, ztx, ztz, 
        xtr, ztr, storage_p, storage_q_1, storage_q_2,
        storage_qq_1, storage_qq_2, storage_qq_3,
        storage_qp
    )
end


"""
blblmmModel
blblmmModel
BLB linear mixed model, which contains a vector of 
`blblmmObs` as data, model parameters, and working arrays.
"""
struct blblmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{blblmmObs{T}}
    w::Vector{T}      # a vector of weights from bootstraping the subset
    # ntotal::Int     # total number of clusters
    p::Int            # number of fixed effect parameters
    q::Int            # number of random effect parameters
    # model parameters
    β::Vector{T}     # fixed effects
    σ²::Vector{T}    # error variance
    Σ::Matrix{T}     # covariance of random effects
    # grad and hess
    ΣL::Matrix{T}
    ∇β::Vector{T} 
    ∇σ²::Vector{T}
    ∇L::Matrix{T}
    Hββ::Matrix{T}   
    Hσ²σ²::Vector{T}
    Hσ²L::Vector{T}
    HLL::Matrix{T}
    # storage
    xtx::Matrix{T}
    xty::Vector{T}
    ztz2::Matrix{T}
    ztr2::Vector{T}
    # the diag indices of L
    diagidx::Vector{Int64}
end

function blblmmModel(obsvec::Vector{blblmmObs{T}}) where T <: BlasReal
    n, p, q = length(obsvec), size(obsvec[1].X, 2), size(obsvec[1].Z, 2)
    q◺ = ◺(q)
    npar = p + 1 + (q * (q + 1)) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the qxq random effect mx has (q * (q + 1))/2 values
    # the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    # then there is the error variance
    w      = ones(T, n) # initialize weights to be 1
    β      = Vector{T}(undef, p)
    σ²     = Vector{T}(undef, 1)
    Σ      = Matrix{T}(undef, q, q)
    ΣL     = Matrix{T}(undef, q, q)
    ∇β     = Vector{T}(undef, p)
    ∇σ²    = Vector{T}(undef, 1)
    ∇L     = Matrix{T}(undef, q, q)
    Hββ    = Matrix{T}(undef, p, p)
    Hσ²σ²  = Vector{T}(undef, 1)
    Hσ²L   = Vector{T}(undef, q◺)
    HLL    = Matrix{T}(undef, q◺, q◺)
    xtx    = Matrix{T}(undef, p, p)
    xty    = Vector{T}(undef, p)
    ztz2    = Matrix{T}(undef, abs2(q), abs2(q))
    ztr2    = Vector{T}(undef, abs2(q))
    diagidx = diag_idx(q)
    # ntotal = 0
    blblmmModel{T}(
        obsvec, w, p, q, 
        β, σ², Σ, ΣL, 
        ∇β, ∇σ², ∇L, Hββ, Hσ²σ², Hσ²L, HLL,
        xtx, xty, ztz2, ztr2, diagidx
    ) 
end


include("lmm.jl")
include("blb.jl")
include("multivariate_calculus.jl")

end # module
