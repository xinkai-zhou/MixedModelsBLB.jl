

"""
    blb_one_subset(y, X, Z, id, N; n_boots, solver, verbose)

Performs Bag of Little Bootstraps on a subset. 

# Positional arguments 
- `y`: response vector
- `X`: design matrix for fixed effects
- `Z`: design matrix for random effects
- `id`: cluster identifier
- `N`: total number of clusters

# Keyword arguments
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `verbose`: print extra information ???

# Values
- `β̂`: a matrix of size n_boots-by-p
- `Σ̂`: a matrix of size n_boots-by-q, which saves the diagonals of Σ̂
- `σ̂2`: a vector of size n_boots
"""
function blb_one_subset(
    # positional arguments
    lmm::LinearMixedModel{Float64},
    y::Vector{T}, 
    X::Matrix{T}, # includes the intercept
    Z::Matrix{T}, 
    id::Vector{Int64},
    N::Int64;
    # keyword arguments
    n_boots::Int64 = 1000,
    solver = Ipopt.IpoptSolver(),
    MoM_init = false,
    verbose::Bool = false
    ) where T <: BlasReal 

    # y = lmm.y, 
    # X = lmm.X, 
    # Z = copy(transpose(first(lmm.reterms).z))
    # writedlm("subset-X.csv", X, ',')
    # writedlm("subset-Z.csv", Z, ',')
    # writedlm("subset-id.csv", id, ',')

    b, p, q = length(Set(id)), size(X, 2), size(Z, 2)
    
    # Initialize arrays for storing the results
    β̂ = Matrix{Float64}(undef, n_boots, p)
    # print("initialized β̂ =", β̂, "\n")
    # print("size of β̂ =", size(β̂), "\n")
    Σ̂ = Matrix{Float64}(undef, n_boots, q) # only save the diagonals
    σ̂2 = zeros(0)

    # β̂ = [Vector{Float64}(undef, p) for i = 1:n_boots]
    # # If we use fill(Vector{Float64}(undef, p), s*r), then all elements of this vector
    # # refer to the same empty vector. So if we use copyto!() to update, all elements of 
    # # the vector will be updated with the same value.

    # Initialize a vector of the blblmmObs objects
    obs = Vector{blblmmObs{Float64}}(undef, b)
    @inbounds @views for (i, grp) in enumerate(unique(id))
        gidx = id .== grp
        yi = Float64.(y[gidx])
        Xi = Float64.(X[gidx, :])
        Zi = Float64.(Z[gidx, :])
        obs[i] = blblmmObs(yi, Xi, Zi)
    end
    # Construct the blblmmModel type
    m = blblmmModel(obs) 
    
    # Initialize arrays for storing subset estimates
    β_b = similar(m.β)
    Σ_b = similar(m.Σ)
    σ2_b = similar(m.σ2)
    
    # # Testing MoM initialization
    # print("Initialize with MoM and print results\n")
    # init_MoM!(m)
    # print("MoM m.β = ", m.β, "\n")
    # print("MoM m.σ2[1] = ", m.σ2[1], "\n")
    # print("MoM m.Σ = ", m.Σ, "\n\n\n")
    # MixedModels.fit!(lmm)
    # copyto!(m.β, lmm.β)
    # m.σ2[1] = 1 / (lmm.σ^2)
    # m.Σ .= Diagonal([i^2 for i in lmm.sigmas[1]]) 
    # print("Initialize with LMM and print results\n")
    # print("LMM m.β = ", m.β, "\n")
    # print("LMM m.σ2[1] = ", m.σ2[1], "\n")
    # print("LMM m.Σ = ", m.Σ, "\n\n\n")

    # Initalize parameters
    if MoM_init 
        # Method of Moments initialization
        init_MoM!(m) # This updates β, σ2 and Σ
    else
        # use MixedModels.jl to initialize
        # fit model
        MixedModels.fit!(lmm)
        copyto!(m.β, lmm.β)
        m.σ2[1] = 1 / (lmm.σ^2)
        extract_Σ!(m.Σ, lmm) 
        # print("m.Σ", m.Σ, "\n")
        # m.Σ .= Diagonal([i^2 for i in lmm.sigmas[1]]) # initialize Σ
    end
    
    # print("After initilization,\n")
    # print("m.β = ", m.β, "\n")
    # print("m.σ2[1] = ", m.σ2[1], "\n")
    # print("m.Σ = ", m.Σ, "\n")
    
    # Fit LMM using the subsample and get parameter estimates
    # fit!(m; solver = solver) 
    # will remove this later because this should be exactly the same as MixedModels.fit!
    copyto!(β_b, m.β)
    copyto!(Σ_b, m.Σ)
    copyto!(σ2_b, m.σ2)
    # print("finished fitting on the subset \n")
    # print("β_b = ", β_b, "\n")
    # print("Σ_b = ", Σ_b, "\n")

    # print("After fitting on the subset,\n")
    # print("m.β = ", m.β, "\n")
    # print("m.σ2[1] = ", m.σ2[1], "\n")
    # print("m.Σ = ", m.Σ, "\n")

    
    ns = zeros(b) # Initialize an array for storing multinomial counts
    re_storage = zeros(q) # for storing random effects
    re_dist = MvNormal(zeros(q), Σ_b) # dist of random effects
    err_dist = Normal(T(0), sqrt(1 / σ2_b[1]))
    mult_prob = ones(b) / b
    mult_dist = Multinomial(N, mult_prob)
    
    bootstrap_runtime = Vector{Float64}()
    # Bootstrapping
    @inbounds for k = 1:n_boots
        verbose && print("Bootstrap iteration ", k, "\n")
        # time0 = time_ns()
        # Generate a parametric bootstrap sample of y and update m.y 
        # in place by looping over blblmmModel.
        @inbounds @views for bidx = 1:b
            rand!(err_dist, m.data[bidx].y) # y = standard normal error
            BLAS.gemv!('N', T(1), m.data[bidx].X, β_b, T(1), m.data[bidx].y) # y = Xβ + error
            rand!(re_dist, re_storage) # simulating random effects
            BLAS.gemv!('N', T(1), m.data[bidx].Z, re_storage, T(1), m.data[bidx].y) # y = Xβ + Zα + error
            # old code
            # copyto!(
            #     m.data[bidx].y, 
            #     m.data[bidx].X * β_b + # fixed effect
            #     m.data[bidx].Z * rand!(mvn_dist, rand_effects) + # random effect
            #     rand(err_dist, length(m.data[bidx].y)) # error, standard normal
            # )
        end
        # # save the data
        # y = Vector{Float64}()
        # for bidx = 1:b
        #     append!(y, m.data[bidx].y)
        # end
        # writedlm("bootstrap-y.csv", y, ',')

        # Get weights by drawing N i.i.d. samples from multinomial
        rand!(mult_dist, ns) 
        # writedlm("bootstrap-ns.csv", ns, ',')

        # Update weights in blblmmModel
        update_w!(m, ns)
        
        # print("Inside bootstrap, before fitting \n")
        # print("m.β = ", m.β, "\n")
        # print("m.σ2[1] = ", m.σ2[1], "\n")
        # print("m.Σ = ", m.Σ, "\n")

        # print("before fit!(),", loglikelihood!(m, false, false), "\n")
        # Use weighted loglikelihood to fit the bootstrapped dataset
        fit!(m; solver = solver)
        
        # print("Inside bootstrap, after fitting\n")
        # print("m.β = ", m.β, "\n")
        # print("m.σ2[1] = ", m.σ2[1], "\n")
        # print("m.Σ = ", m.Σ, "\n")

        # extract estimates
        β̂[k, :] .= m.β 
        # if the assignment is for certain rows of a matrix, then ".=" works fine and we don't need copyto()
        Σ̂[k, :] .= diag(m.Σ)
        push!(σ̂2, m.σ2[1])

        # reset model parameter to subset estimates because 
        # using the bootstrap estimates from each iteration may be unstable.
        copyto!(m.β, β_b)
        copyto!(m.Σ, Σ_b)
        copyto!(m.σ2, σ2_b)
        # m.σ2[1] = σ2_b[1]
        # push!(bootstrap_runtime, (time_ns() - time0)/1e9)
        # print("bootstrap_runtime = ", bootstrap_runtime, "\n")
    end
    return β̂, Σ̂, σ̂2
end



# """
#     blb_full_data(y, X, Z, id, N; subset_size, n_subsets, n_boots, solver, verbose)

# Performs Bag of Little Bootstraps on the full dataset. This interface is intended for smaller datasets that can be loaded in memory.


# # Positional arguments 
# - `y`: response vector
# - `X`: design matrix for fixed effects
# - `Z`: design matrix for random effects
# - `id`: cluster identifier

# # Keyword arguments
# - `subset_size`: number of clusters in the subset. Default to the square root of the total number of clusters.
# - `n_subsets`: number of subsets.
# - `n_boots`: number of bootstrap iterations. Default to 1000
# - `solver`: solver for the optimization problem. 
# - `verbose`: print extra information ???

# # Values
# - `β̂`: a vector (of size n_subsets) of matrices (of size n_boots-by-p)
# - `Σ̂`: a vector (of size n_subsets) of matrices (of size n_boots-by-q, only saves the diagonals of Σ̂)
# - `σ2̂`: a vector (of size n_subsets) of vectors (of size n_boots)
# """

# function blb_full_data(
#     # positional arguments
#     df::DataFrame,
#     f::FormulaTerm;
#     # y::Vector{T}, 
#     # X::Matrix{T}, 
#     # Z::Matrix{T}, 
#     # id::Vector{Int64};
#     # keyword arguments
#     subset_size::Int64 = floor(sqrt(length(unique(id)))),
#     n_subsets::Int64 = 10,
#     n_boots::Int64 = 1000,
#     # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
#     solver = Ipopt.IpoptSolver(),
#     verbose::Bool = false
#     ) where T <: BlasReal 

#     p, q = size(X, 2), size(Z, 2)
#     N = length(unique(id))
#     # print("p = ", p, "\n")
#     # Initialize arrays for storing the results
#     # !! maybe use three dimensional arrays to avoid ugly subsetting    
#     β̂ = [Vector{Float64}(undef, p) for i = 1:(n_subsets * subset_size)]
#     Σ̂ = [Matrix{Float64}(undef, q, q) for i = 1:(n_subsets * subset_size)]
#     σ2̂ = zeros(0)

#     blb_id = fill(0, subset_size)
    
#     Threads.@threads  for j = 1:n_subsets
#         sample!(id, blb_id; replace = false)
#         sort!(blb_id)

#         df_subset = df[subset_indices, :]
#         categorical!(df, Symbol("id"))
#         m = LinearMixedModel(f, df)

#         β̂[((j-1) * n_boots + 1):(j * n_boots)], 
#         Σ̂[((j-1) * n_boots + 1):(j * n_boots)], 
#         σ2̂[((j-1) * n_boots + 1):(j * n_boots)] = 
#         blb_one_subset(
#             # need to implement
#             id[id .== blb_id],
#             N = N;
#             n_boots = n_boots,
#             solver = solver,
#             verbose = verbose
#         )
#     end
#     return β̂, Σ̂, σ2̂
# end



"""
    blb_full_data(file, f; id_name, cat_names, subset_size, n_subsets, n_boots, solver, verbose)

Performs Bag of Little Bootstraps on the full dataset. This interface is intended for larger datasets that cannot fit in memory.

# Positional arguments 
- `file`: file/folder path.
- `f`: model formula.

# Keyword arguments
- `id_name`: name of the cluster identifier variable. String.
- `cat_names`: a vector of the names of the categorical variables.
- `subset_size`: number of clusters in the subset. Default to the square root of the total number of clusters.
- `n_subsets`: number of subsets.
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `verbose`: 

# Values
- `β̂`: a vector (of size n_subsets) of matrices (of size n_boots-by-p)
- `Σ̂`: a vector (of size n_subsets) of matrices (of size n_boots-by-q, only saves the diagonals of Σ̂)
- `σ̂2`: a vector (of size n_subsets) of vectors (of size n_boots)
"""
function blb_full_data(
    # positional arguments
    # !!! could be a folder
    file::String,
    f::FormulaTerm;
    # keyword arguments
    id_name::String,
    cat_names::Vector{String},
    subset_size::Int64,
    n_subsets::Int64 = 10,
    n_boots::Int64 = 1000,
    MoM_init = false,
    solver = Ipopt.IpoptSolver(),
    verbose::Bool = false
    ) where T <: BlasReal 

    # Get variable names from the formula
    lhs_name = [string(x) for x in StatsModels.termvars(f.lhs)]
    rhs_name = [string(x) for x in StatsModels.termvars(f.rhs)]
    var_name = Tuple(x for x in vcat(lhs_name, rhs_name)) # var names need to be in tuples for using select()
    
    # Connect to the dataset/folder
    ftable = JuliaDB.loadtable(
        file, 
        datacols = filter(x -> x != nothing, vcat(lhs_name, rhs_name))
    )

    # By chance, certain factors may not show up in a subset. 
    # To make sure this does not happen, we first 
    # count the number of levels of the categorical variables in the full data,
    # then for each sampled subset, we check whether the number of levels match. 
    # If they do, great. Otherwise, the subset is resampled.
    if length(cat_names) > 0
        cat_levels = Dict{String, Int32}()
        @inbounds for cat_name in cat_names
            cat_levels[cat_name] = length(unique(JuliaDB.select(ftable, Symbol(cat_name))))
        end
    end
    

    # The size of the parameter vectors can only be decided after creating the LinearMixedModel type, 
    # because that's when we recode categorical vars.
    # Initialize arrays for storing the results
    β̂ = Vector{Matrix{Float64}}(undef, n_subsets)
    # Previous initialization: [Vector{Float64}(undef, p) for i = 1:(n_subsets * subset_size)]
    Σ̂ = Vector{Matrix{Float64}}(undef, n_subsets) 
    σ̂2 = Vector{Vector{Float64}}(undef, n_subsets) 
    
    # Load the id column
    id = JuliaDB.select(ftable, Symbol(id_name))
    id_unique = unique(id)
    N = length(id_unique)
    # Initialize an array to store the unique IDs for the subset
    blb_id_unique = fill(0, subset_size)

    # timer = zeros(n_subsets+1)
    # timer[1] = time_ns()
    # Threads.@threads for j = 1:n_subsets
    @inbounds for j = 1:n_subsets
        # https://julialang.org/blog/2019/07/multithreading

        # Count the total number of observations in the subset.
        # n_obs = 0
        # for key in blb_id_unique
        #     n_obs += id_counts[key]
        # end
        # intercept = fill(1., n_obs)

        subset_good = false
        local subset_indices # declare local so that subset_indices is visible after the loop
        while !subset_good
            # Sample from the full dataset
            sample!(id_unique, blb_id_unique; replace = false)
            subset_indices = LinearIndices(id)[findall(in(blb_id_unique), id)]

            if length(cat_names) > 0
                # Count the number of levels of the categorical variables
                cat_levels_subset = Dict{String, Int32}()
                for cat_name in cat_names
                    cat_levels_subset[cat_name] = length(unique(JuliaDB.select(ftable[subset_indices, ], Symbol(cat_name))))
                end
                # If the subset levels do not match the full dataset levels, 
                # skip the current iteration and re-draw blb_id_unique
                if cat_levels_subset == cat_levels
                    subset_good = true
                end
            else 
                subset_good = true
            end
        end
        # Create the LinearMixedModel object using the subset
        # !!! using @views
        # m = LinearMixedModel(f, ftable[subset_indices, ])

        # To use LinearMixedModel(), 
        # we need to transform the id column in ftable to categorical type.
        # Currently I do this by converting the table to DataFrame.
        # In later releases of MixedModels.jl, they will no longer require "id"
        # to be cateogorical. I will change the script then.
        @views df = DataFrame(ftable[subset_indices, ])
        categorical!(df, Symbol("id"))
        lmm = LinearMixedModel(f, df)

        # print("m.X", m.X, "\n")
        # return from blb_one_subset(), a matrix
        β̂[j], Σ̂[j], σ̂2[j] = blb_one_subset(
            lmm,
            lmm.y, 
            lmm.X, 
            copy(transpose(first(lmm.reterms).z)), 
            id[subset_indices],
            N; 
            n_boots = n_boots, 
            MoM_init = MoM_init,
            solver = solver, 
            verbose = verbose
        )
        ####
        # y = select(ftable, Symbol(y_name))[subset_indices, ]
        # # if fe is missing, X is just intercept, otherwise 
        # isnothing(fe_name) ? X = intercept : X = hcat(intercept, JuliaDB.select(ftable, map(Symbol, fe_name))[subset_indices, ])
        # isnothing(re_name) ? Z = intercept : Z = hcat(intercept, JuliaDB.select(ftable, map(Symbol, re_name))[subset_indices, ])
        ####
        # j += 1
        # timer[j] = time_ns()
    end
    return β̂, Σ̂, σ̂2
end



"""
summary(x::Vector{Matrix{Float64}})

# Positional arguments 
- `x`: parameter estimates, a vector (of size n_subsets) of matrices (of size n_boots-by-k), where k is the number of parameters.

# Values
- 
"""
# function summary(x::Vector{Matrix{T}}) where T <: BlasReal 
#     n_subsets = length(x)
#     d = size(x[1], 2)
#     est_all = zeros(d)
#     ci = fill(0., (d, 2))
#     for i = 1 : d
#         est[i] = ??
#         ci[i, :] = StatsBase.percentile(par[((i - 1) * r + 1) : ((i - 1) * r + r)], [2.5, 97.5])
#     end
# end






