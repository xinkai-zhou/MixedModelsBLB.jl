module simulation

println()
@info "Running test2.jl"

using DelimitedFiles
using MixedModelsBLB

# println()
# @info "simulate dataset"
# # Simulate dataset
# Random.seed!(1)
# N = 500 # number of individuals
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
# β̂, Σ̂, τ̂ = blb_one_subset(y, X, Z, id, N)


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
# y = 1 .+ # fixed intercept
#     repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
#     rand(Normal(0, 1), reps * N); # error, standard normal
# x1 = fill(1., reps * N);
# x2 = repeat(vcat(repeat(["m"], 5), repeat(["f"], 5)), 250)
# Z = fill(1., reps * N);
# id = repeat(1:N, inner = reps);
# dat = DataFrame(y=y, x1=x1, x2=x2, Z=Z, id=id)
# CSV.write("testfile.csv", dat)


println()
@info "Test blb_full_data(file, f; id_name, cat_names, subset_size, n_subsets, n_boots, solver, verbose)"
β̂, Σ̂, τ̂ = blb_full_data(
    "testfile.csv", @formula(y ~ 1 + x2 + (1 | id)); 
    id_name = "id", cat_names = ["x2"], subset_size = 300,
    n_subsets = 2, n_boots = 5
)

# writedlm("beta-hat.csv", β̂, ',')
# writedlm("sigma-hat.csv", Σ̂, ',')
# writedlm("tau-hat.csv", τ̂, ',')

end
