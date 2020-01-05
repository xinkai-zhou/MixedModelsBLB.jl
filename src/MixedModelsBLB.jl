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
using Convex
using DataFrames

using LinearAlgebra: BlasReal, copytri!

@reexport using Ipopt
@reexport using NLopt
@reexport using MixedModels

export blblmmObs, blblmmModel
export fit!, fitted, init_β!, loglikelihood!, standardize_res!
export update_res!, update_Σ!, update_w!
export print_matrices
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
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Matrix{T}   # gradient wrt Σ 
    Hβ::Matrix{T}   # Hessian wrt β
    Hτ::Matrix{T}   # Hessian wrt τ
    HΣ::Matrix{T}   # Hessian wrt Σ
    res::Vector{T}  # residual vector
    xtx::Matrix{T}  # Xi'Xi (p-by-p)
    ztz::Matrix{T}  # Zi'Zi (q-by-q)
    xtz::Matrix{T}  # Xi'Zi (p-by-q)
    storage_n1::Vector{T}
    storage_qn::Matrix{T}
    V::Matrix{T}
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
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    ztz = transpose(Z) * Z
    xtz = transpose(X) * Z
    storage_n1 = Vector{T}(undef, n)
    storage_qn = Matrix{T}(undef, q, n)
    V = Matrix{T}(undef, n, n) 
    blblmmObs{T}(y, X, Z, 
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, HΣ,
        res, xtx, ztz, xtz,
        storage_n1, storage_qn, V)
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
    ntotal::Int     # total number of clusters
    p::Int          # number of mean parameters in linear regression
    q::Int          # number of random effects
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter 
    Σ::Matrix{T}    # q-by-q (psd) matrix
    # working arrays
    ΣL::Matrix{T}
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Matrix{T}
    Hβ::Matrix{T}   # Hessian from all observations
    Hτ::Matrix{T}
    HΣ::Matrix{T}
    XtX::Matrix{T}      # X'X = sum_i Xi'Xi
    storage_qq::Matrix{T}
    storage_nq::Matrix{T}
end

function blblmmModel(obsvec::Vector{blblmmObs{T}}) where T <: BlasReal
    n, p, q = length(obsvec), size(obsvec[1].X, 2), size(obsvec[1].Z, 2)
    npar = p + 1 + (q * (q + 1)) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the qxq random effect mx has (q * (q + 1))/2 values
    # the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    # then there is the error variance
    w   = ones(T, n) # initialize weights to be 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    Σ   = Matrix{T}(undef, q, q)
    ΣL  = similar(Σ)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    XtX = zeros(T, p, p) # sum_i xi'xi
    ntotal = 0
    for i in eachindex(obsvec)
        ntotal  += length(obsvec[i].y)
        XtX    .+= obsvec[i].xtx
    end
    storage_qq = Matrix{T}(undef, q, q)
    storage_nq = Matrix{T}(undef, n, q)
    
    blblmmModel{T}(obsvec, w, ntotal, p, q, 
        β, τ, Σ, ΣL,
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, HΣ, 
        XtX, storage_qq, storage_nq)
end


include("lmm.jl")
include("blb.jl")

end # module
