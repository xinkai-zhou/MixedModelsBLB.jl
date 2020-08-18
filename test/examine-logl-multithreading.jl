

Threads.nthreads()

using LinearAlgebra, Random, Distributions
using MixedModelsBLB, BenchmarkTools, Profile
using KNITRO

N = 10^2
reps = 20000
p = 20 # number of fixed effects
q = 2 # number of random effects
βtrue  = ones(p)
σ²true = 1.5
σtrue  = sqrt(σ²true)
Σtrue  = [1.0 0; 0 3.0] # Matrix(Diagonal([2.0; 1.2; 1.0]))
Ltrue  = Matrix(cholesky(Symmetric(Σtrue)).L)
fenames = vcat("Intercept", "x" .* string.([1:1:(p-1);]))
renames = ["Intercept", "z1"]


# generate data
obsvec = Vector{blblmmObs{Float64}}(undef, N)
@inbounds for i in 1:N
    # initialize arrays
    X = Matrix{Float64}(undef, reps, p)
    X[:, 1] = ones(reps)
    Z = Matrix{Float64}(undef, reps, q)
    Z[:, 1] = ones(reps)
    storage_q = Vector{Float64}(undef, q)
    re_storage = Vector{Float64}(undef, q)
    y = Vector{Float64}(undef, reps)
    randn!(y) # y = standard normal error
    @views randn!(X[:, 2:p])
    BLAS.gemv!('N', 1., X, βtrue, σtrue, y) # y = Xβ + σtrue * standard normal error
    randn!(storage_q)
    BLAS.gemv!('N', 1., Ltrue, storage_q, 0., re_storage)
    @views randn!(Z[:, 2:q]) #Distributions.rand!(Normal(), Z[:, 2:q])
    BLAS.gemv!('N', 1., Z, re_storage, 1., y) # y = Xβ + Zα + error
    # y = X * βtrue .+ Z * (Ltrue * randn(q)) .+ σtrue * randn(ns[i])
    obsvec[i] = blblmmObs(y, X, Z)
end

# construct two models 
m1 = blblmmModel(obsvec, fenames, renames, N, true);
m2 = blblmmModel(obsvec, fenames, renames, N, false);

# solver = KNITRO.KnitroSolver(outlev=0)
solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, mehrotra_algorithm = "yes", warm_start_init_point = "yes", warm_start_bound_push = 1e-9)

@benchmark MixedModelsBLB.fit!($m1, solver) setup = init_ls!(m1)
# median time:      6.535 s

@benchmark MixedModelsBLB.fit!($m2, solver) setup = init_ls!($m2)
# median time:      685.603 ms


function update_logl_multithreaded!(
    m::blblmmModel{T}, 
    needgrad::Bool = false, 
    needhess::Bool = false
    ) where T <: BlasReal
    Threads.@threads for obs in m.data  
        loglikelihood!(obs, m.β, m.σ², m.ΣL, needgrad, needhess)
    end
end
function update_logl!(
    m::blblmmModel{T}, 
    needgrad::Bool = false, 
    needhess::Bool = false
    ) where T <: LinearAlgebra.BlasReal
    for obs in m.data  
        loglikelihood!(obs, m.β, m.σ², m.ΣL, needgrad, needhess)
    end
end

BLAS.set_num_threads(2)
@benchmark MixedModelsBLB.update_logl_multithreaded!($m1, false, false) setup = init_ls!($m1)
# median time:      1.271 ms 
BLAS.set_num_threads(1)
@benchmark MixedModelsBLB.update_logl_multithreaded!($m1, false, false) setup = init_ls!($m1)
# median time:      1.913 ms

Profile.clear()
@profile MixedModelsBLB.update_logl_multithreaded!(m1, false, false)
Profile.print(format = :flat)

BLAS.set_num_threads(2)
@benchmark update_logl!($m1, false, false) setup = init_ls!($m1)
# median time:      99.543 μs
BLAS.set_num_threads(1)
@benchmark update_logl!($m1, false, false) setup = init_ls!($m1)
# median time:      98.884 μs

Profile.clear()
@profile update_logl!(m2, false, false)
Profile.print(format = :flat)

# # This is the bottle neck
# @profile for (i, id) in enumerate(subset_id)
#     obsvec[i] = datatable_cols |> 
#         TableOperations.filter(x -> Tables.getcolumn(x, :id) == id) |> 
#         Tables.columns |> 
#         myobs(feformula, reformula)
# end
# Profile.print(format = :flat)