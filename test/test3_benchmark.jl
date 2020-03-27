module blbBenchmark

using MixedModelsBLB, CSV, MixedModels, DataFrames, LinearAlgebra
using Random, Distributions
using Profile, InteractiveUtils, BenchmarkTools, Test



@info "generate data"
Random.seed!(1)
N = 1 # number of individuals
reps = 2000 # number of observations from each individual
x1 = rand(Normal(0, 1), reps * N)
x2 = rand(Normal(0, 3), reps * N)
x3 = rand(Normal(0, 1), reps * N)
X = hcat(x1,x2,x3)
y = 1 .+ # fixed intercept
    x1 + x2 + x3 + 
    repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand(Normal(0, 1), reps * N); # error, standard normal
Z = reshape(fill(1., reps * N), reps * N, 1);
id = repeat(1:N, inner = reps);

# Assign some reasonable starting point
obs = Vector{blblmmObs{Float64}}(undef, N)
for (i, grp) in enumerate(unique(id))
    gidx = id .== grp
    yi = Float64.(y[gidx])
    Xi = Float64.(X[gidx, :])
    Zi = Float64.(Z[gidx, :])
    obs[i] = blblmmObs(yi, Xi, Zi)
end
m = blblmmModel(obs);
m.β .= 0.5 * ones(3)
m.τ .= 0.8
m.Σ .= 0.64
m.ΣL .= 0.8;


@testset "loglikelihood!" begin
@info "benchmark"
bm = @benchmark loglikelihood!($m.data[1], $m.β, $m.τ, $m.Σ, $m.ΣL, false)
display(bm); println()
bm = @benchmark loglikelihood!($m.data[1], $m.β, $m.τ, $m.Σ, $m.ΣL, true)
display(bm); println()
# @test allocs(bm) == 0
bm = @benchmark loglikelihood!($m, true, false)
display(bm); println()
# @test allocs(bm) == 0
# @info "profile"
# Profile.clear()
# @profile @btime mom_obj!($vlmm, true, true)
# Profile.print(format=:flat)
end


end