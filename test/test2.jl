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


println()
@info "Test blb_full_data(file, f)"

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


Random.seed!(1234)

# β̂, Σ̂, τ̂ = blb_full_data(
#     "data/testfile.csv", 
#     @formula(y ~ 1 + x1 + x2 + (1 | id)); 
#     id_name = "id", 
#     cat_names = ["x2"], 
#     subset_size = 300,
#     n_subsets = 2, 
#     n_boots = 3,
#     MoM_init = false,
#     # solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000),
#     # solver = Ipopt.IpoptSolver(print_level = 0),
    # solver = Ipopt.IpoptSolver(
    #     print_level = 5, 
    #     derivative_test = "first-order", 
    #     derivative_test_print_all = "yes"
    # ),
#     # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
#     # solver = Ipopt.IpoptSolver(print_level = 0),
#     verbose = true
# )


β̂_blb, Σ̂_blb, τ̂_blb = blb_full_data(
        # "data/exp2-N-1000-rep-20.csv", 
        "data/exp2-N-10000-rep-20.csv", 
        @formula(y ~ 1 + x1 + x2 + (1 + x1 | id)); 
        id_name = "id", 
        cat_names = Array{String,1}(), 
        subset_size = Int64(floor(10000^0.8)),
        n_subsets = 10, 
        n_boots = 5,
        MoM_init = false,
        # solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000),
        # solver = Ipopt.IpoptSolver(),
        solver = Ipopt.IpoptSolver(
          print_level = 5, 
          derivative_test = "first-order", 
          derivative_test_print_all = "yes"
        ),
        verbose = true
)

@info "Confidence intervals:"
# @show 

writedlm("data/beta-hat.csv", β̂_blb, ',')
writedlm("data/sigma-hat.csv", Σ̂_blb, ',')
writedlm("data/tau-hat.csv", τ̂_blb, ',')
# timer_blb .= timer_blb ./ 1e9
# writedlm("data/timer.csv", timer_blb, ',')

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