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

# CSV.write("test/data/files/File1.csv", dat[1:2000, :])
# CSV.write("test/data/files/File2.csv", dat[2001:4000, :])
# CSV.write("test/data/files/File3.csv", dat[4001:6000, :])
# CSV.write("test/data/files/File4.csv", dat[6001:8000, :])
# CSV.write("test/data/files/File5.csv", dat[8001:10000, :])


result = blb_full_data(
        MersenneTwister(1),
        dat; 
        feformula = @formula(y ~ 1 + x1 + x2 + x3),
        reformula = @formula(y ~ 1 + x1),
        id_name = "id", 
        cat_names = ["x3"], #Array{String,1}(), 
        subset_size = 200, 
        n_subsets = 5, 
        n_boots = 100,
        solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, mehrotra_algorithm = "yes", warm_start_init_point = "yes", warm_start_bound_push = 1e-9),
        verbose = false, 
        nonparametric_boot = true
)
mean_β = fixef(result)
mean_Σ, mean_σ² = vc(result)
ci_β, ci_Σ, ci_σ² = confint(result)

@testset "BLB Result" begin
    @test mean_β[1] ≈ 1.059654143986634 atol = 1e-5
    @test mean_Σ[1, 1] ≈ 1.0470306668380711 atol = 1e-5
    @test mean_σ² ≈ 1.0062658479377924 atol = 1e-5
    
    @test ci_β[1, 1] ≈ 0.9969731108403922 atol = 1e-5
    @test ci_Σ[1, 1] ≈ 0.9661250850647901 atol = 1e-5
    @test ci_σ²[1, 1] ≈ 0.9747415666101469 atol = 1e-5
end

end