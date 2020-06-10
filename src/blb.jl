

"""
SubsetEstimates
BLB linear mixed model estimates from one subset
"""
struct SubsetEstimates{T <: LinearAlgebra.BlasReal}
    # blb parameter
    n_boots::Int64
    # dimension of parameter estimates, for pre-allocation
    p::Int64
    q::Int64
    # blb results
    βs::Vector{Vector{T}}
    Σs::Vector{Matrix{T}}
    σ²s::Vector{T}
end

# constructor
function SubsetEstimates(n_boots::Int64, p::Int64, q::Int64)
    βs  = [Vector{Float64}(undef,  p) for _ in 1:n_boots] #Vector{Vector{Float64}}()
    Σs  = [Matrix{Float64}(undef, q, q) for _ in 1:n_boots] #Vector{Matrix{Float64}}()
    σ²s = Vector{Float64}(undef, n_boots)
    SubsetEstimates(n_boots, p, q, βs, Σs, σ²s)
end

"""
blbEstimates
BLB linear mixed model estimates, which contains 
blb parameters and a vector of `SubsetEstimates`
"""
struct blbEstimates{T <: LinearAlgebra.BlasReal}
    # blb parameters
    n_subsets::Int64
    n_boots::Int64
    # subset estimates from all subsets
    all_estimates::Vector{SubsetEstimates{T}}
end

"""
    save_bootstrap_result!(subset_estimates, β, Σ, σ²)

Save the result from one bootstrap iteration to subset_estimates

# Positional arguments 
- `subset_estimates`: an object of the SubsetEstimates type
- `i`: update the ith element of βs, Σs and σ²s
- `β`: parameter estimates for fixed effects
- `Σ`: the covariance matrix of variance components
- `σ²`: estimate of error variance 
"""
function save_bootstrap_result!(
    subset_estimates::SubsetEstimates{T},
    i::Int64,
    β::Vector{T},
    Σ::Matrix{T},
    σ²::T
    ) where T <: BlasReal
    subset_estimates.βs[i] = β
    subset_estimates.Σs[i] = Σ
    subset_estimates.σ²s[i] = σ²
end


"""
    blb_one_subset(m; n_boots, solver, verbose)

Performs Bag of Little Bootstraps on a subset. 

# Positional arguments 
- `m`: an object of the blblmmModel type

# Keyword arguments
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `verbose`: Bool, whether to print bootstrap progress (percentage completion)

# Values
- `subset_estimates`: an object of the SubsetEstimates type
"""
function blb_one_subset(
    # positional arguments
    m::blblmmModel{T};
    # keyword arguments
    n_boots::Int64 = 1000,
    solver = Ipopt.IpoptSolver(),
    verbose::Bool = false
    ) where T <: BlasReal 

    # Initalize model parameters
    init_ls!(m)
    
    # Fit LMM on the subset
    fit!(m; solver = solver) 

    # Initalize an instance of SubsetEstimates type for storing results
    subset_estimates = SubsetEstimates(n_boots, m.p, m.q)

    # construct the simulator type
    simulator = Simulator(m)

    # Bootstrapping
    @inbounds for k = 1:n_boots
        verbose && print("Bootstrap iteration ", k, "\n")

        # Parametric bootstrapping. Updates m.data[i].y for all i
        simulate!(m, simulator)

        # Get weights by drawing N i.i.d. samples from multinomial
        rand!(simulator.mult_dist, simulator.ns) 
        # print("simulator.ns[1:10] = ", simulator.ns[1:10], "\n")
        
        # Update weights in blblmmModel
        update_w!(m, simulator.ns)
        
        # Fit model on the bootstrap sample
        fit!(m; solver = solver)
        # print("m.β = ", m.β, "\n")
        # print("m.Σ = ", m.Σ, "\n")
        
        # Save estimates
        save_bootstrap_result!(subset_estimates, k, m.β, m.Σ, m.σ²[1])

        # Reset model parameter to subset estimates as initial parameters because
        # using the bootstrap estimates from each iteration may be unstable.
        copyto!(m.β, simulator.β_subset)
        copyto!(m.Σ, simulator.Σ_subset)
        copyto!(m.σ², simulator.σ²_subset)
    end
    return subset_estimates
end


"""
    blblmmobs(datatable)

Construct the blblmmObs type

# Positional arguments 
- `data_obs`: a table object that is compatible with Tables.jl
- `feformula`: the formula for the fixed effects
- `reformula`: the formula for the fixed effects

# Values
- `cat_levels`: a dictionary that contains the number of levels of each categorical variable.
"""
function blblmmobs(data_obs::Union{Tables.AbstractColumns, DataFrames.DataFrame}, feformula::FormulaTerm, reformula::FormulaTerm)
    y, X = StatsModels.modelcols(feformula, data_obs)
    Z = StatsModels.modelmatrix(reformula, data_obs)
    return blblmmObs(y, X, Z)
end
blblmmobs(feformula::FormulaTerm, reformula::FormulaTerm) = data_obs -> blblmmobs(data_obs, feformula, reformula)

"""
    count_levels(data_columns, cat_names)

Count the number of levels of each categorical variable.

# Positional arguments 
- `data_columns`: an object of the AbstractColumns type
- `cat_names`: a character vector of the names of the categorical variables

# Values
- `cat_levels`: a dictionary that contains the number of levels of each categorical variable.
"""
function count_levels(data_columns::Union{Tables.AbstractColumns, DataFrames.DataFrame}, cat_names::Vector{String})
    cat_levels = Dict{String, Int64}()
    @inbounds for cat_name in cat_names
        cat_levels[cat_name] = length(countmap(Tables.getcolumn(data_columns, Symbol(cat_name))))
    end
    return cat_levels
end
count_levels(cat_names::Vector{String}) = data_columns -> count_levels(data_columns, cat_names)

"""
    subsetting!(subset_id, data_columns, id_name, unique_id, cat_names, cat_levels)

Draw a subset from the full dataset.

# Positional arguments 
- `subset_id`: a vector for storing the IDs of the subset
- `data_columns`: an object of the AbstractColumns type, or a DataFrame
- `unique_id`: a vector of the unique ID in the full data set
- `cat_names`: a character vector of the names of the categorical variables
- `cat_levels`: a dictionary that contains the number of levels of each categorical variable
"""
function subsetting!(
    subset_id::Vector,
    # data_columns::Union{Tables.AbstractColumns, DataFrame},
    data_columns,
    id_name::Symbol,
    unique_id::Vector,
    cat_names::Vector{String},
    cat_levels::Dict
    )
    good_subset = false
    while !good_subset
        # Sample from the full dataset
        sample!(unique_id, subset_id; replace = false)
        # subset_indices = LinearIndices(id)[findall(in(blb_id_unique), id)]
        if length(cat_names) > 0
            cat_levels_subset = data_columns |> 
                TableOperations.filter(x -> Tables.getcolumn(x, Symbol(id_name)) .∈ Ref(Set(subset_id))) |> 
                Tables.columns |> 
                count_levels(cat_names)
            # If the subset levels do not match the full dataset levels, 
            # skip the current iteration and take another subset
            if cat_levels_subset == cat_levels
                good_subset = true
            end
        else 
            good_subset = true
        end
    end
end


"""
    blb_full_data(file, f; id_name, cat_names, subset_size, n_subsets, n_boots, solver, verbose)

Performs Bag of Little Bootstraps on the full dataset. This interface is intended for larger datasets that cannot fit in memory.

# Positional arguments 
- `datatable`: a data table type that is compatible with Tables.jl

# Keyword arguments
- `feformula`: model formula for the fixed effects.
- `reformula`: model formula for the random effects.
- `id_name`: name of the cluster identifier variable. String.
- `cat_names`: a vector of the names of the categorical variables.
- `subset_size`: number of clusters in the subset. Default to the square root of the total number of clusters.
- `n_subsets`: number of subsets.
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `verbose`: Bool, whether to print bootstrap progress (percentage completion)

# Values
- `result`: an object of the blbEstimates type
"""
function blb_full_data(
    # positional arguments
    datatable;
    # keyword arguments
    feformula::FormulaTerm,
    reformula::FormulaTerm,
    id_name::String,
    cat_names::Vector{String} = Vector{String}(),
    subset_size::Int64,
    n_subsets::Int64 = 10,
    n_boots::Int64 = 1000,
    solver = Ipopt.IpoptSolver(),
    verbose::Bool = false
    )

    typeof(id_name) <: String && (id_name = Symbol(id_name))

    # apply df-wide schema
    feformula = apply_schema(feformula, schema(feformula, datatable))
    reformula = apply_schema(reformula, schema(reformula, datatable))
    
    fenames = coefnames(feformula)[2]
    renames = coefnames(reformula)[2]

    # By chance, some factors of a categorical variable may not show up in a subset. 
    # To make sure this does not happen, we
    # 1. count the number of levels of each categorical variable in the full data;
    # 2. for each sampled subset, we check whether the number of levels match. 
    # If they do not match, the subset is resampled.
    datatable_cols = Tables.columns(datatable)
    if length(cat_names) > 0
        cat_levels = count_levels(datatable_cols, cat_names)
    else
        cat_levels = Dict()
    end

    # Get the unique ids, which will be used for subsetting
    unique_id = unique(Tables.getcolumn(datatable_cols, id_name))
    N = length(unique_id) # number of individuals/clusters in the full dataset
    # Initialize an array to store the unique IDs for the subset
    subset_id = Vector{eltype(unique_id)}(undef, subset_size)

    # Initialize a vector of SubsetEstimates for storing estimates from subsets
    all_estimates = Vector{SubsetEstimates{Float64}}(undef, n_subsets)

    # Initialize a vector of the blblmmObs objects
    # ??? Float64 or some other type?
    obsvec = Vector{blblmmObs{Float64}}(undef, subset_size)

    # Threads.@threads for j = 1:n_subsets
    @inbounds for j = 1:n_subsets
        # https://julialang.org/blog/2019/07/multithreading
        
        # Take a subset
        subsetting!(subset_id, datatable_cols, id_name, unique_id, cat_names, cat_levels)
        
        # Construct blblmmObs objects
        @inbounds for (i, id) in enumerate(subset_id)
            obsvec[i] = datatable_cols |> 
                TableOperations.filter(x -> Tables.getcolumn(x, id_name) == id) |> 
                Tables.columns |> 
                blblmmobs(feformula, reformula)
        end

        # Construct the blblmmModel type
        m = blblmmModel(obsvec, fenames, renames, N) 

        # Run BLB on this subset
        all_estimates[j] = blb_one_subset(
            m;
            n_boots = n_boots, 
            solver = solver, 
            verbose = verbose
        )
    end

    # Create a blbEstimates instance for storing results from all subsets
    result = blbEstimates{Float64}(n_subsets, n_boots, all_estimates)
    return result
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
# - `σ²̂`: a vector (of size n_subsets) of vectors (of size n_boots)
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
#     σ²̂ = zeros(0)

#     blb_id = fill(0, subset_size)
    
#     Threads.@threads  for j = 1:n_subsets
#         sample!(id, blb_id; replace = false)
#         sort!(blb_id)

#         df_subset = df[subset_indices, :]
#         categorical!(df, Symbol("id"))
#         m = LinearMixedModel(f, df)

#         β̂[((j-1) * n_boots + 1):(j * n_boots)], 
#         Σ̂[((j-1) * n_boots + 1):(j * n_boots)], 
#         σ²̂[((j-1) * n_boots + 1):(j * n_boots)] = 
#         blb_one_subset(
#             # need to implement
#             id[id .== blb_id],
#             N = N;
#             n_boots = n_boots,
#             solver = solver,
#             verbose = verbose
#         )
#     end
#     return β̂, Σ̂, σ²̂
# end




