
module Fit

# Simulate a dataset and test functions: loglikelihood!() and fit!()
using MixedModelsBLB, LinearAlgebra, Random, Distributions, Test

Random.seed!(123)
N = 100; p = 4; q = 2; reps = 5
βtrue  = [0.1; 6; -3; 1]
σ²true = 1.5
σtrue  = sqrt(σ²true)
Σtrue  = [1.0 0; 0 3.0] # Matrix(Diagonal([2.0; 1.2; 1.0]))
Ltrue  = Matrix(cholesky(Symmetric(Σtrue)).L)
obsvec = Vector{blblmmObs{Float64}}(undef, N)
X = Matrix{Float64}(undef, reps, p)
X[:, 1] = ones(reps)
Z = Matrix{Float64}(undef, reps, q)
Z[:, 1] = ones(reps)
storage_q = Vector{Float64}(undef, q)
re_storage = Vector{Float64}(undef, q)
y = Vector{Float64}(undef, reps)
fenames = vcat("Intercept", "x" .* string.([1:1:(p-1);]))
renames = ["Intercept", "z1"]

# Generate data
for i in 1:N
    randn!(y) # y = standard normal error
    # first column intercept, remaining entries iid std normal
    @views randn!(X[:, 2:p]) #Distributions.rand!(Normal(), X[:, 2:p])
    BLAS.gemv!('N', 1., X, βtrue, σtrue, y) # y = Xβ + σtrue * standard normal error
    randn!(storage_q)
    BLAS.gemv!('N', 1., Ltrue, storage_q, 0., re_storage)
    # first column intercept, remaining entries iid std normal
    @views randn!(Z[:, 2:q]) #Distributions.rand!(Normal(), Z[:, 2:q])
    BLAS.gemv!('N', 1., Z, re_storage, 1., y) # y = Xβ + Zα + error
    # y = X * βtrue .+ Z * (Ltrue * randn(q)) .+ σtrue * randn(ns[i])
    # form a blblmmObs instance
    obsvec[i] = blblmmObs(copy(y), copy(X), copy(Z))
end

# form a LmmModel instance
lmm = blblmmModel(obsvec, fenames, renames, N)
obsN = lmm.data[N]

# Data was simulated correctly
@test obsN.y[1] ≈ -13.248956453781018 atol = 1e-5

@testset "loglikelihood!()" begin
    @test loglikelihood!(obsN, βtrue, [σ²true], Ltrue, true, true) ≈ -10.753503805549846 atol = 1e-5
    
    init_ls!(lmm)
    @test loglikelihood!(lmm, true, true) ≈ -1090.2939385079767 atol = 1e-5
end

@testset "fit!()" begin
    fit!(lmm)
    @test loglikelihood!(lmm, true, false) ≈ -994.0524142638102 atol = 1e-5

    @test lmm.β[1] ≈ 0.20516618221633012 atol = 1e-5

    @test norm(lmm.∇β) < 1e-5
    @test norm(lmm.∇σ²) < 1e-5
    @test norm(lmm.∇L) < 1e-5
end