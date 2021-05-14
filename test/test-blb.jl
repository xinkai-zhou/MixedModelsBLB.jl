module BLB

# Test that the BLB functions work properly
using MixedModelsBLB, Random, Distributions, DataFrames, StatsModels, Ipopt, Test

# Simulate a dataset
# βtrue  = ones(3); σ²true = 1; Σtrue  = [1.0 0; 0 1.0]
Random.seed!(123)
N = 1000; reps = 10
x1 = rand(Normal(0, 1), reps * N)
x2 = rand(Normal(0, 1), reps * N)
x3 = repeat(["M", "F"], inner = reps * N >> 1)
rand_slope = zeros(reps * N)
for j in 1:N
    rand_slope[(reps * (j-1) + 1) : reps * j] = x1[(reps * (j-1) + 1) : reps * j] .* rand(Normal(0, 1), 1)
end
y = 1 .+ x1 + x2 + # fixed effects
    repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand_slope +
    rand(Normal(0, 1), reps * N) # error, standard normal
id = repeat(1:N, inner = reps)
dat = DataFrame(y = y, x1 = x1, x2 = x2, x3 = x3, id = id)

result = blb_full_data(
        MersenneTwister(1),
        dat; 
        feformula = @formula(y ~ 1 + x1 + x2 + x3),
        reformula = @formula(y ~ 1 + x1),
        id_name = "id", 
        cat_names = ["x3"], #Array{String,1}(), 
        subset_size = 200, 
        n_subsets = 1, 
        n_boots = 100,
        solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, mehrotra_algorithm = "yes", warm_start_init_point = "yes", warm_start_bound_push = 1e-9),
        verbose = false, 
        nonparametric_boot = true
)
mean_β = fixef(result)
mean_Σ, mean_σ² = vc(result)
ci_β, ci_Σ, ci_σ² = confint(result)

@testset "BLB Result" begin
    # Since RNG is not stable between Julia versions, the tests below can fail on any version.
    # One solution is to use StableRNG https://github.com/JuliaRandom/StableRNGs.jl
    # Otherwise we just don't test the actual numbers. Instead, just test n_boots, n_subsets etc.
    
    # @test mean_β[1] ≈ 1.0826426652008734 atol = 1e-5
    # @test mean_Σ[1, 1] ≈ 1.0517803598655269 atol = 1e-5
    # @test mean_σ² ≈ 1.033196835173124 atol = 1e-5
    
    # @test ci_β[1, 1] ≈ 0.9862575001544613 atol = 1e-5
    # @test ci_Σ[1, 1] ≈ 0.9495329320935163 atol = 1e-5
    # @test ci_σ²[1, 1] ≈ 1.0042431857583207 atol = 1e-5

    @test result.subset_size == 200
    @test result.n_subsets == 1
    @test result.n_boots == 100
end

end
