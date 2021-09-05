"""
    NonparametricBootSimulator

The NonparametricBootSimulator type holds the parameters and storages for simulating the multinomial counts
"""
struct NonparametricBootSimulator{T <: LinearAlgebra.BlasReal}
    # # estimated obtained from the subset
    # β_subset::Vector{T}
    # Σ_subset::Matrix{T}
    # σ²_subset::Vector{T}
    # for simulating multinomial counts
    ns::Vector{Int64}
    mult_prob::Vector{T}
    mult_dist::Distributions.Multinomial 
end
    
"""
    NonparametricBootSimulator(m)

The constructor for the `NonparametricBootSimulator` type.

# Keyword arguments
- `N`: number of individuals/clusters in the full dataset.
- `subset_size`: number of individuals/clusters in each subset.   
"""
function NonparametricBootSimulator(
    m::Union{blblmmModel{T}, WSVarLmmModel{T}};
    N::Int64, 
    subset_size::Int64
    ) where T <: BlasReal
    # initialize a vector for storing multinomial counts
    ns = zeros(Int64, subset_size) 
    mult_prob = ones(subset_size) / subset_size
    mult_dist = Multinomial(N, mult_prob)
    # NonparametricBootSimulator(β_subset, Σ_subset, σ²_subset, ns, mult_prob, mult_dist)
    NonparametricBootSimulator(ns, mult_prob, mult_dist)
end


"""
    ParametricBootSimulator

The `ParametricBootSimulator` type holds the parameters and storages for simulating the response and the multinomial counts
"""
struct ParametricBootSimulator{T <: LinearAlgebra.BlasReal}
    # estimated obtained from the subset
    β_subset::Vector{T}
    Σ_subset::Matrix{T}
    ΣL_subset::Matrix{T}
    σ²_subset::Vector{T}
    # for simulating the response y
    Xβ::Vector{Vector{T}}
    storage_q::Vector{T}
    re_storage::Vector{T}
    # re_dist::MvNormal
    # for simulating multinomial counts
    ns::Vector{Int64}
    mult_prob::Vector{T}
    mult_dist::Distributions.Multinomial 
end
    
"""
    ParametricBootSimulator(m)

The constructor for the `ParametricBootSimulator` type.

# Keyword arguments
- `N`: number of individuals/clusters in the full dataset.
- `subset_size`: number of individuals/clusters in each subset.   
"""
function ParametricBootSimulator(
    m::blblmmModel{T};
    N::Int64, 
    subset_size::Int64
    ) where T <: BlasReal
    # *_subset will not change throughout bootstrap iterations
    β_subset = similar(m.β)
    copyto!(β_subset, m.β)
    Σ_subset = similar(m.Σ)
    copyto!(Σ_subset, m.Σ)
    ΣL_subset = similar(m.ΣL) #Matrix(cholesky(Symmetric(Σ_subset)).L)
    copyto!(ΣL_subset, m.ΣL)
    σ²_subset = similar(m.σ²)
    copyto!(σ²_subset, m.σ²)
    
    # Since xtβ is constant throughout simulation, we pre-compute it
    Xβ = Vector{Vector{T}}(undef, subset_size)
    @inbounds @views for i in 1:subset_size
        Xβ[i] = Vector{T}(undef, m.data[i].n)
        BLAS.gemv!('N', T(1), m.data[i].X, β_subset, T(0), Xβ[i])
    end

    # initialize a vector for storing α 
    re_storage = Vector{T}(undef, m.q)
    storage_q = Vector{T}(undef, m.q)
    # distribution of the random effect 
    # re_dist = MvNormal(zeros(m.q), Σ_subset)
    # initialize a vector for storing multinomial counts
    ns = zeros(Int64, subset_size) 
    mult_prob = ones(subset_size) / subset_size
    mult_dist = Multinomial(N, mult_prob)
    ParametricBootSimulator(
        β_subset, Σ_subset, ΣL_subset, σ²_subset, 
        Xβ, storage_q, re_storage,
        ns, mult_prob, mult_dist
        )
end


function simulate!(
    rng::Random.AbstractRNG, 
    m::blblmmModel{T},
    simulator::ParametricBootSimulator{T}
    ) where T<: LinearAlgebra.BlasReal
    σ = sqrt(simulator.σ²_subset[1])
    @inbounds @views for bidx = 1:length(simulator.ns)
        randn!(rng, m.data[bidx].y) # y = standard normal error
        BLAS.axpby!(T(1), simulator.Xβ[bidx], σ, m.data[bidx].y) # y = Xβ + σ * standard normal error
        # simulate random effect: ΣL * standard normal
        randn!(rng, simulator.storage_q)
        BLAS.gemv!('N', 1., simulator.ΣL_subset, simulator.storage_q, 0., simulator.re_storage)
        BLAS.gemv!('N', T(1), m.data[bidx].Z, simulator.re_storage, T(1), m.data[bidx].y) # y = Xβ + Zα + error

        # Update quantities related to y in blblmmObs
        m.data[bidx].yty[1] = dot(m.data[bidx].y, m.data[bidx].y)
        BLAS.gemv!('T', T(1), m.data[bidx].X, m.data[bidx].y, T(0), m.data[bidx].xty)
        BLAS.gemv!('T', T(1), m.data[bidx].Z, m.data[bidx].y, T(0), m.data[bidx].zty)
        # copyto!(m.data[bidx].xty, transpose(m.data[bidx].X) * m.data[bidx].y)
        # copyto!(m.data[bidx].zty, transpose(m.data[bidx].Z) * m.data[bidx].y)
    end
end
