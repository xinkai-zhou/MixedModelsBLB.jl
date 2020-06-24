"""
Simulator

The Simulator type holds the parameters and storages for 
simulating the response and the multinomial counts
"""
struct Simulator{T <: LinearAlgebra.BlasReal}
    m::MixedModelsBLB.blblmmModel{T}
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
    
# Constructor
"""
    Simulator(m, b, N)

The constructor for  the Simulator type.

"""
function Simulator(
    m::blblmmModel{T},
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
    Xβ = Vector{Vector{T}}(undef, m.b)
    @inbounds for i in 1:m.b
        Xβ[i] = m.data[i].y
        BLAS.gemv!('N', T(1), m.data[i].X, β_subset, T(0), Xβ[i])
    end
    # initialize a vector for storing α 
    re_storage = Vector{T}(undef, m.q)
    storage_q = Vector{T}(undef, m.q)
    # distribution of the random effect 
    # re_dist = MvNormal(zeros(m.q), Σ_subset)
    # initialize a vector for storing multinomial counts
    ns = zeros(Int64, m.b) 
    mult_prob = ones(m.b) / m.b
    mult_dist = Multinomial(m.N, mult_prob)
    Simulator(
        m, 
        β_subset, Σ_subset, ΣL_subset, σ²_subset, 
        Xβ, storage_q, re_storage,
        ns, mult_prob, mult_dist
        )
end


function simulate!(
    # rng::Random.AbstractRNG, 
    m::MixedModelsBLB.blblmmModel{T},
    simulator::Simulator{T}
    ) where T<: LinearAlgebra.BlasReal
    σ = sqrt(simulator.σ²_subset[1])
    @inbounds @views for bidx = 1:m.b
        randn!(m.data[bidx].y) # y = standard normal error
        BLAS.axpby!(T(1), simulator.Xβ[bidx], σ, m.data[bidx].y) # y = Xβ + σ * standard normal error
        # simulate random effect: ΣL * standard normal
        randn!(simulator.storage_q)
        BLAS.gemv!('N', 1., simulator.ΣL_subset, simulator.storage_q, 0., simulator.re_storage)
        BLAS.gemv!('N', T(1), m.data[bidx].Z, simulator.re_storage, T(1), m.data[bidx].y) # y = Xβ + Zα + error
    end
end
