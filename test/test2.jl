module simulation

println()
@info "Running test2.jl"

using Random, Distributions, LinearAlgebra, DelimitedFiles
using MixedModelsBLB

println()
@info "simulate dataset"
# Simulate dataset
Random.seed!(1)
N = 500 # number of individuals
p = 1
q = 1
reps = 5 # number of observations from each individual
y = 1 .+ # fixed intercept
    repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand(Normal(0, 1), reps * N); # error, standard normal
X = fill(1., (reps * N, p));
Z = fill(1., (reps * N, q));
id = repeat(1:N, inner = reps);

println()
@info "Take a subset to test blb_one_subset()"

β̂, Σ̂, τ̂ = blb_one_subset(y, X, Z, id)

end
