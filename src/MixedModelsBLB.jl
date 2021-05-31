__precompile__()

module MixedModelsBLB

using MathProgBase
using Reexport
using Distributions
using Random
using StatsModels
using StatsBase
using DataFrames
using InteractiveUtils
using Permutations
using LinearAlgebra
using Tables
using TableOperations
using Distributed
using Query
using DBInterface

using LinearAlgebra: BlasReal, copytri!
import LinearAlgebra: BlasFloat, checksquare

@reexport using Ipopt
@reexport using NLopt

export blblmmObs, blblmmModel
export update_w!, init_ls!, fit!, loglikelihood! # lmm.jl
export SubsetEstimates, blbEstimates, save_bootstrap_result!, blb_one_subset, blb_full_data, blb_db # blb.jl
export confint, fixef, vc, coeftable # blb.jl
export Simulator, simulate! # simulate.jl
export ◺ # multivariate_calculus.jl

"""
    blblmmObs

BLB linear mixed model observation type. Contains data from a single cluster, working arrays and so on.
"""
struct blblmmObs{T <: LinearAlgebra.BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T} # X should include a column of 1's
    Z::Matrix{T}
    n::Int
    # grad and hess
    ∇β::Vector{T}   # gradient wrt β
    ∇σ²::Vector{T}   # gradient wrt σ²
    ∇L::Matrix{T}   # gradient wrt L 
    Hββ::Matrix{T}   # Hessian wrt β
    Hσ²σ²::Vector{T}   # Hessian wrt σ²
    Hσ²L::Vector{T}   # Hessian cross term
    HLL::Matrix{T}   # Hessian wrt L
    # pre-compute
    yty::Vector{T} 
    xty::Vector{T} 
    zty::Vector{T}
    xtx::Matrix{T}  # Xi'Xi (p-by-p)
    ztx::Matrix{T}  # Zi'Xi (q-by-p)
    ztz::Matrix{T}  # Zi'Zi (q-by-q)
    # working arrays
    # obj::Vector{T}
    xtr::Vector{T}
    ztr::Vector{T}
    storage_p::Vector{T}
    storage_q_1::Vector{T}
    storage_q_2::Vector{T}
    storage_qq_1::Matrix{T}
    storage_qq_2::Matrix{T}
    storage_qq_3::Matrix{T}
    storage_qp::Matrix{T}
    
    # # data
    # y          :: Vector{T}
    # X          :: Matrix{T}
    # Z          :: Matrix{T}
    # # gradient
    # ∇β         :: Vector{T}
    # ∇σ²        :: Vector{T}
    # ∇Σ         :: Matrix{T}
    # # Hessian
    # Hββ        :: Matrix{T}
    # HLL        :: Matrix{T}
    # Hσ²σ²      :: Matrix{T}
    # Hσ²L       :: Matrix{T}
    # Hσ²Lvec       :: Vector{T}
    # # TODO: add whatever intermediate arrays you may want to pre-allocate
    # yty        :: T
    # xty        :: Vector{T}
    # zty        :: Vector{T}
    # ztr        :: Vector{T}
    # ltztr      :: Vector{T}
    # xtr        :: Vector{T}
    # storage_p  :: Vector{T}
    # storage_q  :: Vector{T}
    # xtx        :: Matrix{T}
    # ztx        :: Matrix{T}
    # ztz        :: Matrix{T}
    # ltztzl     :: Matrix{T}
    # lminvlt    :: Matrix{T}
    # M          :: Matrix{T}
    # ztΩinvz    :: Matrix{T}
    # storage_qq :: Matrix{T}
    # storage_pq :: Matrix{T}
end

"""
    blblmmObs(y, X, Z)

The constructor for the blblmmObs type.

# Positional arguments 
- `y`: response vector
- `X`: design matrix for fixed effects
- `Z`: design matrix for random effects.
"""
function blblmmObs(
    y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T}
    ) where T <: BlasReal
    n, p = size(X)
    q = size(Z, 2)
    q◺ = ◺(q)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    ∇β     = Vector{T}(undef, p)
    ∇σ²    = Vector{T}(undef, 1)
    ∇L     = Matrix{T}(undef, q, q)
    Hββ    = Matrix{T}(undef, p, p)
    Hσ²σ²  = Vector{T}(undef, 1)
    Hσ²L   = Vector{T}(undef, q◺)
    HLL    = Matrix{T}(undef, q◺, q◺)
    # obj = Vector{T}(undef, 1)
    yty = [dot(y, y)]
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
        n,
        ∇β, ∇σ², ∇L, 
        Hββ, Hσ²σ², Hσ²L, HLL,
        yty, xty, zty, xtx, ztx, ztz, 
        # obj,
        xtr, ztr, storage_p, storage_q_1, storage_q_2,
        storage_qq_1, storage_qq_2, storage_qq_3,
        storage_qp
    )

    # y::Vector{T}, 
    # X::Matrix{T}, 
    # Z::Matrix{T}) where T <: AbstractFloat
    # n, p, q    = size(X, 1), size(X, 2), size(Z, 2) 
    # ∇β         = Vector{T}(undef, p)
    # ∇σ²        = Vector{T}(undef, 1)
    # ∇Σ         = Matrix{T}(undef, q, q)
    # Hββ        = Matrix{T}(undef, p, p)
    # HLL        = Matrix{T}(undef, ◺(q), ◺(q))
    # Hσ²σ²      = Matrix{T}(undef, 1, 1)
    # Hσ²L       = Matrix{T}(undef, q, q)
    # Hσ²Lvec    = Vector{T}(undef, ◺(q))
    # yty        = abs2(norm(y))
    # xty        = transpose(X) * y
    # zty        = transpose(Z) * y
    # ztr        = similar(zty)
    # ltztr      = similar(zty)
    # xtr        = Vector{T}(undef, p)
    # storage_p  = similar(xtr)
    # storage_q  = Vector{T}(undef, q)
    # xtx        = transpose(X) * X
    # ztx        = transpose(Z) * X
    # ztz        = transpose(Z) * Z
    # ltztzl     = similar(ztz)
    # lminvlt    = similar(ztz)
    # M          = similar(ztz)
    # ztΩinvz    = similar(ztz)
    # storage_qq = similar(ztz)
    # storage_pq = Matrix{T}(undef, p, q)
    # blblmmObs(y, X, Z, ∇β, ∇σ², ∇Σ, Hββ, HLL, Hσ²σ², Hσ²L, Hσ²Lvec,
    #     yty, xty, zty, ztr, ltztr, xtr,
    #     storage_p, storage_q, 
    #     xtx, ztx, ztz, ltztzl, lminvlt, M, ztΩinvz, 
    #     storage_qq, storage_pq)
end


"""
    blblmmModel

BLB linear mixed model type, which contains a vector of `blblmmObs` as data, model parameters, and working arrays.
"""
struct blblmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{blblmmObs{T}}
    fenames::Vector{String} # a vector of the fixed effect variable names
    renames::Vector{String} # a vector of the random effect variable names
    N::Int    # total number of unique IDs (individuals) in the full data set
    b::Int    # total number of unique IDs (individuals) in the subset
    p::Int            # number of fixed effect parameters
    q::Int            # number of random effect parameters
    q◺::Int           # number of parameters in the cholesky factor of Σ
    w::Vector{Int}      # a vector of weights from bootstraping the subset
    # model parameters
    β::Vector{T}     # fixed effects
    σ²::Vector{T}    # error variance
    Σ::Matrix{T}     # covariance of random effects
    ΣL::Matrix{T}    # lower cholesky factor of Σ
    # grad and hess
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
    # whether to use multi-threading in evaluating the loglikelihood
    # use_threads::Bool
end


"""
    blblmmModel(obsvec)

The constructor for  the blblmmModel type.

# Positional arguments 
- `obsvec`: a vector of type blblmmObs
"""
function blblmmModel(
    obsvec::Vector{blblmmObs{T}},
    fenames::Vector{String},
    renames::Vector{String},
    N::Int64
    # use_threads::Bool
    ) where T <: BlasReal
    # T = eltype(obsvec[1].X)
    b, p, q = length(obsvec), size(obsvec[1].X, 2), size(obsvec[1].Z, 2)
    q◺ = ◺(q)
    npar = p + 1 + (q * (q + 1)) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the qxq random effect mx has (q * (q + 1))/2 values
    # the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    # then there is the error variance
    w      = ones(Int64, b) # initialize weights to be 1
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
        obsvec, fenames, renames, 
        N, b, p, q, q◺, w,
        β, σ², Σ, ΣL, 
        ∇β, ∇σ², ∇L, Hββ, Hσ²σ², Hσ²L, HLL,
        xtx, xty, ztz2, ztr2, diagidx
        # use_threads
    ) 
end


include("lmm.jl")
include("blb.jl")
include("simulate.jl")
include("multivariate_calculus.jl")

end # module
