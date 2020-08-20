

using LinearAlgebra, StatsModels, Random, Distributions, MixedModelsBLB, JuliaDB

# Run blb
feformula   = @formula(y ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + 
                            x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + 
                            x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + 
                            x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + 
                            x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + 
                            x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 + x60 + 
                            x61 + x62 + x63 + x64 + x65 + x66 + x67 + x68 + x69 + x70 + 
                            x71 + x72 + x73 + x74 + x75 + x76 + x77 + x78 + x79 + x80 + 
                            x81 + x82 + x83 + x84 + x85 + x86 + x87 + x88 + x89 + x90 + 
                            x91 + x92 + x93 + x94 + x95 + x96 + x97 + x98 + x99)
reformula   = @formula(y ~ 1 + z1)

# datatable = JuliaDB.loadtable("test/data/testdata-N-20000-REPS-10.csv")
datatable = JuliaDB.loadtable("test/data/testdata-N-20000-REPS-10.csv")
blb_full_data(
        MersenneTwister(1),
        datatable;
        feformula = feformula,
        reformula = reformula,
        id_name = "id", cat_names = Array{String,1}(), 
        subset_size = 1000, n_subsets = 1, n_boots = 10,
        solver = Ipopt.IpoptSolver(print_level=5, max_iter=100, mehrotra_algorithm = "yes", warm_start_init_point = "yes"),
        # solver = Ipopt.IpoptSolver(
        #   print_level = 5, derivative_test = "second-order", derivative_test_print_all = "yes", check_derivatives_for_naninf = "yes"
        # ),
        verbose = true, use_threads = false, use_groupby = true, nonparametric_boot = true
)


# # Generate Data
# # Global parameters
# N = 20000 # number of individuals
# reps = 10 # number of observations from each individual
# # id = repeat(1:N, inner = reps)
# p = 100 # number of fixed effects
# q = 2 # number of random effects
# use_threads = false

# # true parameter values
# βtrue  = ones(100) #[0.1; 6; -3; 1]
# σ²true = 1
# σtrue  = sqrt(σ²true)
# Σtrue  = [1.0 1; 1 4.0] # Matrix(Diagonal([2.0; 1.2; 1.0]))
# Ltrue  = Matrix(cholesky(Symmetric(Σtrue)).L)

# obsvec = Vector{blblmmObs{Float64}}(undef, N)
# # initialize arrays
# X = Matrix{Float64}(undef, reps, p)
# X[:, 1] = ones(reps)
# Z = Matrix{Float64}(undef, reps, q)
# Z[:, 1] = ones(reps)
# storage_q = Vector{Float64}(undef, q)
# re_storage = Vector{Float64}(undef, q)
# y = Vector{Float64}(undef, reps)
# fenames = vcat("Intercept", "x" .* string.([1:1:(p-1);]))
# renames = ["Intercept", "z1"]

# # Generate data
# for i in 1:N
#     randn!(y) # y = standard normal error
#     # first column intercept, remaining entries iid std normal
#     @views randn!(X[:, 2:p]) #Distributions.rand!(Normal(), X[:, 2:p])
#     BLAS.gemv!('N', 1., X, βtrue, σtrue, y) # y = Xβ + σtrue * standard normal error
#     randn!(storage_q)
#     BLAS.gemv!('N', 1., Ltrue, storage_q, 0., re_storage)
#     # first column intercept, remaining entries iid std normal
#     @views randn!(Z[:, 2:q]) #Distributions.rand!(Normal(), Z[:, 2:q])
#     BLAS.gemv!('N', 1., Z, re_storage, 1., y) # y = Xβ + Zα + error
#     # y = X * βtrue .+ Z * (Ltrue * randn(q)) .+ σtrue * randn(ns[i])
#     # form a blblmmObs instance
#     obsvec[i] = blblmmObs(copy(y), copy(X), copy(Z))
# end
# # form a LmmModel instance
# lmm = blblmmModel(obsvec, fenames, renames, N, use_threads)

# # Save the data to a CSV file.
# open(string("test/data/testdata-N-", N, "-REPS-", reps, ".csv"), "w") do io
#     p = size(lmm.data[1].X, 2)
#     q = size(lmm.data[1].Z, 2)
#     # print header
#     print(io, "id,y,")
#     for j in 1:(p-1)
#         print(io, "x" * string(j) * ",")
#     end
#     for j in 1:(q-1)
#         print(io, "z" * string(j) * (j < q-1 ? "," : "\n"))
#     end
#     # print data
#     for i in eachindex(lmm.data)
#         obs = lmm.data[i]
#         for j in 1:length(obs.y)
#             # id
#             print(io, i, ",")
#             # y
#             print(io, obs.y[j], ",")
#             # X data
#             for k in 2:p
#                 print(io, obs.X[j, k], ",")
#             end
#             # Z data
#             for k in 2:q-1
#                 print(io, obs.Z[j, k], ",")
#             end
#             print(io, obs.Z[j, q], "\n")
#         end
#     end
# end

