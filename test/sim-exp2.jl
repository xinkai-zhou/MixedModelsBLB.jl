
# Comparing speed


using MixedModelsBLB, MixedModels, Random, Distributions, DataFrames, CSV, RCall
Random.seed!(1)
# ((Int64(1e4), 20), (Int64(1e4), 50), (Int64(1e4), 20), (Int64(1e4), 50), (Int64(1e5), 20), (Int64(1e5), 50))
datasizes = ((Int64(1e4), 20), (Int64(1e4), 50)) 
#for (N, reps) in datasizes
#    # simulate data
#    x1 = rand(Normal(0, 1), reps * N)
#    x2 = rand(Normal(0, 3), reps * N)
#    rand_slope = zeros(reps * N)
#    @views for j in 1:N
#        rand_slope[(reps * (j-1) + 1) : reps * j] = x1[(reps * (j-1) + 1) : reps * j] .* rand(Normal(0, 2), 1)
#    end
#    y = 1 .+ x1 + x2 + # fixed effects
#        repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
#        rand_slope +
#        rand(Normal(0, 1), reps * N) # error, standard normal
#    id = repeat(1:N, inner = reps)
#    dat = DataFrame(y=y, x1=x1, x2=x2, id=id)
#    CSV.write(string("../data/exp2-N-", N, "-rep-", reps, ".csv"), dat)
#end


# MixedModelsBLB
blb_runtime = Vector{Float64}()
for (N, reps) in datasizes
    time0 = time_ns()
    β̂_blb, Σ̂_blb, τ̂_blb, timer_blb = blb_full_data(
        string("data/exp2-N-", N, "-rep-", rep, ".csv"), 
        @formula(y ~ 1 + x1 + x2 + (1 + x1 | id)); 
        id_name = "id", 
        cat_names = [], 
        subset_size = N^0.6,
        n_subsets = 10, 
        n_boots = 2000,
        MoM_init = false,
        solver = Ipopt.IpoptSolver(),
        verbose = true
    )
    push!(blb_runtime, (time_ns() - time0)/1e9)
end
CSV.write("data/blb_runtime.csv", blb_runtime)

# MixedModels + bootstrap
B = 2000 # number of bootstrap samples
mixedmodels_runtime = Vector{Float64}()
for (N, reps) in datasizes
    time0 = time_ns()
    dat = CSV.read(string("data/exp2-N-", N, "-rep-", rep, ".csv"))
    categorical!(dat, Symbol("id"))
    lmm = LinearMixedModel(@formula(y ~ x1 + x2 + (1 + x1 | id)), dat)
    fit!(lmm)
    const rng = MersenneTwister(1234321);
    boot = parametricbootstrap(rng, B, lmm)
    # Get CI
    push!(mixedmodels_runtime, (time_ns() - time0)/1e9)
end
CSV.write("data/mixedmodels_runtime.csv", mixedmodels_runtime)

# Rcall, lme4 + bootstrap
lme4_runtime = Vector{Float64}()
for (N, reps) in datasizes
    filename = string("data/exp2-N-", N, "-rep-", rep, ".csv")
    time0 = time_ns()
    R"""
    library(lme4)
    dat = read.csv($filename, header = T)
    lmm = lme(y ~ x1 + x2 + (1 + x1 | id)), dat)
    # Summary functions
    mySumm <- function(.) { 
        s <- sigma(.)
        c(beta =getME(., "beta"), sigma = s, sig01 = unname(s * getME(., "theta"))) 
        }
    ## alternatively:
    mySumm2 <- function(.) {
        c(beta=fixef(.),sigma=sigma(.), sig01=sqrt(unlist(VarCorr(.))))
    }
    boo01 <- bootMer(lmm, mySumm, nsim = $B)
    """
    push!(lme4_runtime, (time_ns() - time0)/1e9)
end

CSV.write("data/lme4_runtime.csv", lme4_runtime)























