module simulation

println()
@info "Running test2.jl"

using DelimitedFiles, Random, CSV
using MixedModelsBLB

# println()
# @info "simulate dataset"
# # Simulate dataset
# Random.seed!(1)
# N = 50 # number of individuals
# p = 1
# q = 1
# reps = 5 # number of observations from each individual
# y = 1 .+ # fixed intercept
#     repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
#     rand(Normal(0, 1), reps * N); # error, standard normal
# X = fill(1., (reps * N, p));
# Z = fill(1., (reps * N, q));
# id = repeat(1:N, inner = reps);

# println()
# @info "Test blb_one_subset()"
# # β̂, Σ̂, τ̂ = blb_one_subset(y, X, Z, id, N; n_boots = 10, solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000))
# β̂, Σ̂, τ̂ = blb_one_subset(
#     y, X, Z, id, N; 
#     n_boots = 2,
#     solver = Ipopt.IpoptSolver(
#         print_level = 5, 
#         derivative_test = "first-order", 
#         derivative_test_print_all = "yes"
#     )
# )



# !!!!!!!!!!!!!! Skipped for now.
# println()
# @info "Test blb_full_data(y, X, Z, id, N; subset_size, n_subsets, n_boots, solver, verbose)"
# β̂, Σ̂, τ̂ = blb_full_data(y, X, Z, id, N; n_subsets = 2, n_boots = 100)
# LoadError: LoadError: MethodError: no method matching #blb_full_data#6(::Float64, ::Int64, ::Int64, ::NLopt.NLoptSolver, ::Bool, ::typeof(MixedModelsBLB.blb_full_data), ::Array{Float64,1}, ::Array{Float64,2}, ::Array{Float64,2}, ::Array{Int64,1}, ::Int64)
# Closest candidates are:
  #blb_full_data#6(::Int64, ::Int64, ::Int64, ::Any, ::Bool, ::Any, ::Array{T<:Union{Float32, Float64},1}, ::Array{T<:Union{Float32, Float64},2}, ::Array{T<:Union{Float32, Float64},2}, ::Array{Int64,1}, ::Int64) where T<:Union{Float32, Float64}


# This part was run in the terminal. The output "testfile.csv" was moved to the "test" folder
# Random.seed!(1)
# N = 500 # number of individuals
# reps = 5 # number of observations from each individual
# x1 = rand(Normal(0, 1), reps * N)
# x2 = repeat(vcat(repeat(["m"], 5), repeat(["f"], 5)), 250)
# y = 1 .+ # fixed intercept
#     x1 + 
#     repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
#     rand(Normal(0, 1), reps * N); # error, standard normal

# Z = fill(1., reps * N);
# id = repeat(1:N, inner = reps);
# dat = DataFrame(y=y, x1=x1, x2=x2, Z=Z, id=id)
# CSV.write("testfile.csv", dat)

# using CSV, DataFrames, MixedModels, JuliaDB, StatsModels
# f = @formula(y ~ 1 + x1 + x2 + (1 | id))
# lhs_name = [string(x) for x in StatsModels.termvars(f.lhs)]
# rhs_name = [string(x) for x in StatsModels.termvars(f.rhs)]
# # d = CSV.read("testfile.csv"; header= true)
# # df = DataFrame(d)
# ftable = JuliaDB.loadtable(
#     "testfile.csv", 
#     datacols = filter(x -> x != nothing, vcat(lhs_name, rhs_name))
# )
# subset_indices = [1:1500;]
# df = DataFrame(ftable[subset_indices, ])
# categorical!(df, Symbol("id"))
# lmm = LinearMixedModel(f, df)
# function test(lmm::LinearMixedModel)
#     fit!(lmm)
# end
# propertynames(lmm)
# fit!(lmm)
# propertynames(lmm)

println()
@info "Test blb_full_data(file, f)"

Random.seed!(1234)

β̂, Σ̂, τ̂ = blb_full_data(
    "data/testfile.csv", 
    @formula(y ~ 1 + x1 + x2 + (1 | id)); 
    id_name = "id", 
    cat_names = ["x2"], 
    subset_size = 300,
    n_subsets = 10, 
    n_boots = 20,
    MoM_init = false,
    # solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000),
    # solver = Ipopt.IpoptSolver(print_level = 0),
    solver = Ipopt.IpoptSolver(
        print_level = 5, 
        derivative_test = "first-order", 
        derivative_test_print_all = "yes"
    ),
    # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    # solver = Ipopt.IpoptSolver(print_level = 0),
    verbose = true
)




# β̂, Σ̂, τ̂ = blb_full_data(
#     "testfile.csv", 
#     @formula(y ~ 1 + x1 + x2 + (1 | id)); 
#     id_name = "id", 
#     cat_names = ["x2"], 
#     subset_size = 300,
#     n_subsets = 2, 
#     n_boots = 5,
#     MoM_init = false,
#     solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000)
# )


# testlmm("testfile.csv", @formula(y ~ 1 + x1 + x2 + (1 | id)), "id")
# pkgs = Pkg.installed()
# pkgs["MixedModels"]
# testlmm(
#     "sleepstudy.csv", 
#     @formula(Reaction ~ 1 + Days + (Days | Subject)), 
#     "Subject"
# ) 
writedlm("data/beta-hat.csv", β̂, ',')
writedlm("data/sigma-hat.csv", Σ̂, ',')
writedlm("data/tau-hat.csv", τ̂, ',')

end



# Simulate a large dataset for profiling
# Random.seed!(1)
# N = 500 # number of individuals
# reps = 5 # number of observations from each individual
# x1 = rand(Normal(0, 1), reps * N)
# x2 = repeat(vcat(repeat(["m"], 5), repeat(["f"], 5)), 250)
# y = 1 .+ # fixed intercept
#     x1 + 
#     repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
#     rand(Normal(0, 1), reps * N); # error, standard normal

# Z = fill(1., reps * N);
# id = repeat(1:N, inner = reps);
# dat = DataFrame(y=y, x1=x1, x2=x2, Z=Z, id=id)
# CSV.write("testfile.csv", dat)