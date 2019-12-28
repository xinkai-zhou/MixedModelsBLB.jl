
using Random, Distributions, LinearAlgebra, DelimitedFiles, DataFrames
using MixedModelsBLB

# Simulate data
# simulate y, X, id and form a DataFrame
N = 500 # number of individuals
p = 1
q = 1
reps = 5 # number of observations from each individual

y = 1 .+ # fixed intercept
    repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand(Normal(0, 1), reps * N); # error, standard normal
x = repeat(1:reps, N);
id = repeat(1:N, inner = reps);
dat = DataFrame(y = y, x = x, id = id)

y, X, Z, id = print_matrices(@formula(y ~ 1 + x + (1|id)), dat)

writedlm("y.csv", y, ',')
writedlm("X.csv", X, ',')
writedlm("Z.csv", Z, ',')
writedlm("id.csv", id, ',')
