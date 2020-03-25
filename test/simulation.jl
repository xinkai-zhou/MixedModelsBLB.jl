using MixedModelsBLB, MixedModels, Random, Distributions, DataFrames, CSV


# 1. Get ground truth CI

# 1.1 Simulate 2000 datasets, estimate parameters on each and get CI
r = 2000
β̂_truth = zeros(r, 2)
σ̂_0_truth = zeros(r)
Random.seed!(1)
for i in 1:r
    # simulate dataset
    N = 1e6 # number of individuals
    reps = 20 # number of observations from each individual
    x1 = rand(Normal(0, 1), reps * N)
    x2 = rand(Normal(0, 3), reps * N)
    # X = hcat(x1, x2)
    y = 1 .+ x1 + x2 + # fixed intercept
        repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
        rand(Normal(0, 1), reps * N); # error, standard normal
    # Z = reshape(fill(1., reps * N), reps * N, 1);
    id = repeat(1:N, inner = reps);
    # fit model and extract parameters
    df = DataFrame(y=y, x1=x1, x2=x2, id=id)
    categorical!(df, Symbol("id"))
    lmm = LinearMixedModel(f, df)
    fit!(lmm)
    # Check whether extraction is correct
    β̂_truth[i, :] = lmm.beta
    σ̂_0_truth[i] = lmm.sigma
end

# Simulate a single dataset that will be used by BLB and bootstrap
Random.seed!(1)
N = 1e6 # number of individuals
reps = 20 # number of observations from each individual
x1 = rand(Normal(0, 1), reps * N)
x2 = rand(Normal(0, 3), reps * N)
y = 1 .+ x1 + x2 + # fixed intercept
    repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand(Normal(0, 1), reps * N); # error, standard normal
Z = reshape(fill(1., reps * N), reps * N, 1);
id = repeat(1:N, inner = reps);
dat = DataFrame(y=y, x1=x1, x2=x2, Z=Z, id=id)
CSV.write("exp1-testfile.csv", dat)

# 1.2 MixedModelsBLB 
β̂_blb, Σ̂_blb, τ̂_blb, timer_blb = blb_full_data(
    "data/testfile.csv", 
    @formula(y ~ 1 + x1 + x2 + (1 | id)); 
    id_name = "id", 
    cat_names = [], 
    subset_size = (1e6)^0.6,
    n_subsets = 10, 
    n_boots = 2000,
    MoM_init = false,
    # solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000),
    # solver = Ipopt.IpoptSolver(print_level = 0),
    solver = Ipopt.IpoptSolver(),
    # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    # solver = Ipopt.IpoptSolver(print_level = 0),
    verbose = true
)

# 1.3 MixedModels bootstrap


























