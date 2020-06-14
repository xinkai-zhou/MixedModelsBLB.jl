
# Comparing speed

using MixedModels, Random, Distributions, DataFrames, CSV, JuliaDB
using MixedModelsBLB

# ((Int64(1e4), 20), (Int64(1e4), 50), (Int64(1e4), 20), (Int64(1e4), 50), (Int64(1e5), 20), (Int64(1e5), 50))
# datasizes = ((Int64(1e4), 20), (Int64(1e4), 20)) 
Random.seed!(1)
reps = 20
N = 10000
x1 = rand(Normal(0, 1), reps * N)
x2 = rand(Normal(0, 3), reps * N)
rand_slope = zeros(reps * N)
@views for j in 1:N
    rand_slope[(reps * (j-1) + 1) : reps * j] = x1[(reps * (j-1) + 1) : reps * j] .* rand(Normal(0, 2), 1)
end
y = 1 .+ x1 + x2 + # fixed effects
    repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand_slope +
    rand(Normal(0, 1), reps * N) # error, standard normal
id = repeat(1:N, inner = reps)
dat = DataFrame(y=y, x1=x1, x2=x2, id=id)

# for (N, reps) in datasizes
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
#    CSV.write(string("data/exp2-N-", N, "-rep-", reps, ".csv"), dat)
# end

# blb_runtime = Vector{Float64}()
# for (N, reps) in datasizes
#     time0 = time_ns()
#     β̂_blb, Σ̂_blb, τ̂_blb = blb_full_data(
#         string("data/exp2-N-", N, "-rep-", reps, ".csv"), 
#         @formula(y ~ 1 + x1 + x2 + (1 + x1 | id)); 
#         id_name = "id", 
#         cat_names = Array{String,1}(), 
#         subset_size = N,
#         n_subsets = 1, 
#         n_boots = 10,
#         MoM_init = false,
#         solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, ftol_rel = 1.0e-12, ftol_abs = 1.0e-8, maxeval=10000),
#         # solver = Ipopt.IpoptSolver(print_level = 0),
#         # solver = Ipopt.IpoptSolver(
#         #   print_level = 5, 
#         #   derivative_test = "first-order", 
#         #   derivative_test_print_all = "yes"
#         # ),
#         verbose = true
#     )
#     push!(blb_runtime, (time_ns() - time0)/1e9)
#     print("LN_BOBYQA blb_runtime (in seconds) at N = ", N, ", reps = ", reps, " = ", blb_runtime, "\n")
# end

# dat = loadtable("data/exp2-N-1000-rep-20.csv")
result = blb_full_data(
        dat;
        feformula = @formula(y ~ 1 + x1 + x2),
        reformula = @formula(y ~ 1 + x1),
        id_name = "id", 
        cat_names = Array{String,1}(), 
        subset_size = 1000,
        n_subsets = 5, 
        n_boots = 10,
        solver = Ipopt.IpoptSolver(print_level = 0),
        # solver = Ipopt.IpoptSolver(
        #   print_level = 5, 
        #   derivative_test = "second-order", 
        #   derivative_test_print_all = "yes",
        #   check_derivatives_for_naninf = "yes"
        # ),
        verbose = false
        )
print(result)
# blb_runtime = Vector{Float64}()
# for (N, reps) in datasizes
#     time0 = time_ns()
#     β̂_blb, Σ̂_blb, τ̂_blb = blb_full_data(
#         string("data/exp2-N-", N, "-rep-", reps, ".csv"), 
#         @formula(y ~ 1 + x1 + x2 + (1 + x1 | id)); 
#         id_name = "id", 
#         cat_names = Array{String,1}(), 
#         subset_size = N,
#         n_subsets = 1, 
#         n_boots = 1,
#         MoM_init = false,
#         # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, ftol_rel = 1.0e-12, ftol_abs = 1.0e-8, maxeval=10000),
#         # solver = Ipopt.IpoptSolver(print_level = 0),
#         solver = Ipopt.IpoptSolver(
#           print_level = 5, 
#           derivative_test = "first-order", 
#           derivative_test_print_all = "yes"
#         #   check_derivatives_for_naninf = "yes"
#         ),
#         verbose = true
#     )
#     push!(blb_runtime, (time_ns() - time0)/1e9)
#     print("IPOPT blb_runtime (in seconds) at N = ", N, ", reps = ", reps, " = ", blb_runtime, "\n")
# end



# MixedModels + bootstrap
# B = 10 # number of bootstrap samples
# rng = MersenneTwister(1234321)
# mixedmodels_runtime = Vector{Float64}()
# for (N, reps) in datasizes
#     time0 = time_ns()
#     dat = CSV.read(string("data/exp2-N-", N, "-rep-", reps, ".csv"))
#     categorical!(dat, Symbol("id"))
#     lmm = LinearMixedModel(@formula(y ~ x1 + x2 + (1 + x1 | id)), dat)
#     MixedModels.fit!(lmm)
#     boot = parametricbootstrap(rng, B, lmm)
#     # Get CI
#     push!(mixedmodels_runtime, (time_ns() - time0)/1e9)
#     print("mixedmodels_runtime = ", mixedmodels_runtime, "\n")
# end

# CSV.write("data/mixedmodels_runtime.csv", mixedmodels_runtime)

# # Rcall, lme4 + bootstrap
# lme4_runtime = Vector{Float64}()
# for (N, reps) in datasizes
#     filename = string("data/exp2-N-", N, "-rep-", reps, ".csv")
#     time0 = time_ns()
#     R"""
#     library(lme4)
#     dat = read.csv($filename, header = T)
#     lmm = lmer(y ~ x1 + x2 + (1 + x1 | id), dat)
#     # Summary functions
#     mySumm <- function(.) { 
#         s <- sigma(.)
#         c(beta =getME(., "beta"), sigma = s, sig01 = unname(s * getME(., "theta"))) 
#         }
#     ## alternatively:
#     # mySumm2 <- function(.) {
#     #     c(beta=fixef(.),sigma=sigma(.), sig01=sqrt(unlist(VarCorr(.))))
#     # }
#     boo01 <- bootMer(lmm, mySumm, nsim = $B)
#     """
#     push!(lme4_runtime, (time_ns() - time0)/1e9)
# end
# print("lme4_runtime = ", lme4_runtime, "\n")
# # CSV.write("data/lme4_runtime.csv", lme4_runtime)

# old code

# # MixedModels + bootstrap
# B = 2000 # number of bootstrap samples
# mixedmodels_runtime = Vector{Float64}()
# for (N, reps) in datasizes
#     time0 = time_ns()
#     dat = CSV.read(string("data/exp2-N-", N, "-rep-", reps, ".csv"))
#     categorical!(dat, Symbol("id"))
#     lmm = LinearMixedModel(@formula(y ~ x1 + x2 + (1 + x1 | id)), dat)
#     fit!(lmm)
#     const rng = MersenneTwister(1234321);
#     boot = parametricbootstrap(rng, B, lmm)
#     # Get CI
#     push!(mixedmodels_runtime, (time_ns() - time0)/1e9)
# end
# CSV.write("data/mixedmodels_runtime.csv", mixedmodels_runtime)

# # Rcall, lme4 + bootstrap
# lme4_runtime = Vector{Float64}()
# for (N, reps) in datasizes
#     filename = string("data/exp2-N-", N, "-rep-", reps, ".csv")
#     time0 = time_ns()
#     R"""
#     library(lme4)
#     dat = read.csv($filename, header = T)
#     lmm = lmer(y ~ x1 + x2 + (1 + x1 | id), dat)
#     # Summary functions
#     mySumm <- function(.) { 
#         s <- sigma(.)
#         c(beta =getME(., "beta"), sigma = s, sig01 = unname(s * getME(., "theta"))) 
#         }
#     ## alternatively:
#     mySumm2 <- function(.) {
#         c(beta=fixef(.),sigma=sigma(.), sig01=sqrt(unlist(VarCorr(.))))
#     }
#     boo01 <- bootMer(lmm, mySumm, nsim = $B)
#     """
#     push!(lme4_runtime, (time_ns() - time0)/1e9)
# end

























