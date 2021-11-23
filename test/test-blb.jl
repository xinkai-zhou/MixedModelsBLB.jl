module BLB

# Test that the BLB functions work properly
using MixedModelsBLB, Random, Distributions, DataFrames, StatsModels, Ipopt, Test, StableRNGs, WiSER

# Simulate a dataset
# βtrue  = ones(3); σ²true = 1; Σtrue  = [1.0 0; 0 1.0]
rng = StableRNG(123)
# Random.seed!(rng, 123)
N = 1000; reps = 10
x1 = rand(rng, Normal(0, 1), reps * N)
x2 = rand(rng, Normal(0, 1), reps * N)
x3 = repeat(["M", "F"], inner = reps * N >> 1)
rand_slope = zeros(reps * N)
for j in 1:N
    rand_slope[(reps * (j-1) + 1) : reps * j] = x1[(reps * (j-1) + 1) : reps * j] .* rand(rng, Normal(0, 1), 1)
end
y = 1 .+ x1 + x2 + # fixed effects
    repeat(rand(rng, Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand_slope +
    rand(rng, Normal(0, 1), reps * N) # error, standard normal
id = repeat(1:N, inner = reps)
dat = DataFrame(y = y, x1 = x1, x2 = x2, x3 = x3, id = id)

# Parametric BLB with Maximum Likelihood
result0 = blb_full_data(
        StableRNG(123),
        dat; 
        feformula = @formula(y ~ 1 + x1 + x2 + x3),
        reformula = @formula(y ~ 1 + x1),
        id_name = "id", 
        cat_names = ["x3"], #Array{String,1}(), 
        subset_size = 200, 
        n_subsets = 2, 
        n_boots = 100,
        method = :ML,
        solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, mehrotra_algorithm = "yes", warm_start_init_point = "yes", warm_start_bound_push = 1e-9),
        verbose = false, 
        nonparametric_boot = false
)
mean_β = fixef(result0)
mean_Σ, mean_σ² = vc(result0)
ci_β, ci_Σ, ci_σ² = MixedModelsBLB.confint(result0)

@testset "Parametric BLB Result for Method = :ML" begin
    # Since RNG is not stable between Julia versions, the tests below can fail on any version.
    # One solution is to use StableRNG https://github.com/JuliaRandom/StableRNGs.jl
    # Otherwise we just don't test the actual numbers. Instead, just test n_boots, n_subsets etc.
    
    @test mean_β[1] ≈ 0.9523749104428877 atol = 1e-3
    @test mean_Σ[1, 1] ≈ 1.094385387720442 atol = 1e-3
    @test mean_σ² ≈ 1.0150958802746968 atol = 1e-3
    
    @test ci_β[1, 1] ≈ 0.7406733280497191 atol = 1e-3
    @test ci_Σ[1, 1] ≈ 0.8490824216956047 atol = 1e-3
    @test ci_σ²[1, 1] ≈ 0.9453143174741494 atol = 1e-3
end

# Nonparametric BLB with Maximum Likelihood
result1 = blb_full_data(
        StableRNG(123),
        dat; 
        feformula = @formula(y ~ 1 + x1 + x2 + x3),
        reformula = @formula(y ~ 1 + x1),
        id_name = "id", 
        cat_names = ["x3"], #Array{String,1}(), 
        subset_size = 200, 
        n_subsets = 2, 
        n_boots = 100,
        method = :ML,
        solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, mehrotra_algorithm = "yes", warm_start_init_point = "yes", warm_start_bound_push = 1e-9),
        verbose = false, 
        nonparametric_boot = true
)
mean_β = fixef(result1)
mean_Σ, mean_σ² = vc(result1)
ci_β, ci_Σ, ci_σ² = MixedModelsBLB.confint(result1)

@testset "Nonparametric BLB Result for Method = :ML" begin
    # Since RNG is not stable between Julia versions, the tests below can fail on any version.
    # One solution is to use StableRNG https://github.com/JuliaRandom/StableRNGs.jl
    # Otherwise we just don't test the actual numbers. Instead, just test n_boots, n_subsets etc.
    
    @test mean_β[1] ≈ 0.9681988820743825 atol = 1e-3
    @test mean_Σ[1, 1] ≈ 1.0331613190126188 atol = 1e-3
    @test mean_σ² ≈ 1.011932405830439 atol = 1e-3
    
    @test ci_β[1, 1] ≈ 0.8811621102951823 atol = 1e-3
    @test ci_Σ[1, 1] ≈ 0.9379266586770029 atol = 1e-3
    @test ci_σ²[1, 1] ≈ 0.9776523913354277 atol = 1e-3
end


# Nonparametric BLB with WiSER
result2 = blb_full_data(
        StableRNG(123),
        dat; 
        feformula    = @formula(y ~ 1 + x1 + x2 + x3),
        reformula    = @formula(y ~ 1 + x1),
        wsvarformula = @formula(y ~ 1),
        id_name      = "id", 
        cat_names    = ["x3"], #Array{String,1}(), 
        subset_size  = 200, 
        n_subsets    = 2,
        method       = :WiSER,
        n_boots      = 100,
        solver       = Ipopt.IpoptSolver(print_level=0),
        verbose      = false, 
        nonparametric_boot = true
)
mean_β = fixef(result2)
mean_Σ, mean_σ² = vc(result2)
ci_β, ci_Σ, ci_σ² = MixedModelsBLB.confint(result2)

@testset "Nonparametric BLB Result for Method = :WiSER" begin
    @test mean_β[1] ≈ 0.9682015976735611 atol = 1e-3
    @test mean_Σ[1, 1] ≈ 1.0331762861674212 atol = 1e-3
    @test mean_σ² ≈ 1.0119282472432958 atol = 1e-3
    
    @test ci_β[1, 1] ≈ 0.8811874356556482 atol = 1e-3
    @test ci_Σ[1, 1] ≈ 0.9379372642984771 atol = 1e-3
    @test ci_σ²[1, 1] ≈ 0.9776457919670338 atol = 1e-3
end


end
