

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
    βs::Matrix{T}
    Σs::Array{T}
    σ²s::Vector{T}
end

# constructor
function SubsetEstimates(n_boots::Int64, p::Int64, q::Int64)
    # vector of vector/matrices approach
    # βs  = [Vector{Float64}(undef,  p) for _ in 1:n_boots] #Vector{Vector{Float64}}()
    # Σs  = [Matrix{Float64}(undef, q, q) for _ in 1:n_boots] #Vector{Matrix{Float64}}()

    # multi-dimensional array approach
    βs  = Matrix{Float64}(undef, n_boots, p) 
    Σs  = Array{Float64}(undef, q, q, n_boots) # 3d array 
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
    subset_size::Int64
    n_boots::Int64
    fenames::Vector{String}
    renames::Vector{String}
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
    # using the vector of vectors/matrices approach
    # copyto!(subset_estimates.βs[i], β)
    # copyto!(subset_estimates.Σs[i], Σ)

    # using the multi-dimensional array approach
    subset_estimates.βs[i, :] = β
    subset_estimates.Σs[:, :, i] = Σ
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
    m::blblmmModel{T};
    n_boots::Int64 = 1000,
    solver = Ipopt.IpoptSolver(),
    verbose::Bool = false
    ) where T <: BlasReal 

    # Initalize model parameters
    init_ls!(m, verbose)
    
    # Fit LMM on the subset
    fit!(m, solver)
    verbose && print("m.Σ = ", m.Σ, "\n") 

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
        fit!(m, solver)
        # print("After fitting on the bootstrap sample, m.β = ", m.β, "\n")
        # print("m.Σ = ", m.Σ, "\n")
        
        # Save estimates
        save_bootstrap_result!(subset_estimates, k, m.β, m.Σ, m.σ²[1])
        # print("After save_bootstrap_result, subset_estimates = ", subset_estimates, "\n")

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
function blblmmobs(data_obs, feformula::FormulaTerm, reformula::FormulaTerm)
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
function count_levels(data_columns, cat_names::Vector{String})
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
    result = blbEstimates{Float64}(n_subsets, subset_size, n_boots, fenames, renames, all_estimates)
    return result
end



function confint(subset_ests::SubsetEstimates, level::Real)
    ci_βs = Matrix{Float64}(undef, subset_ests.p, 2) # p-by-2 matrix
    for i in 1:subset_ests.p
        ci_βs[i, :] = StatsBase.percentile(view(subset_ests.βs, :, i), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
    end
    ◺q = ◺(subset_ests.q)
    ci_Σs = Matrix{Float64}(undef, ◺q, 2)
    k = 1
    # For Σ, we get the CI for the upper-triangular values.
    @inbounds for i in 1:subset_ests.q
        @inbounds for j in i:subset_ests.q
            ci_Σs[k, :] = StatsBase.percentile(view(subset_ests.Σs, i, j, :), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
            k += 1
        end
    end
    ci_σ²s = reshape(StatsBase.percentile(subset_ests.σ²s, 100 * [(1 - level) / 2, 1 - (1-level) / 2]), 1, 2)
    return ci_βs, ci_Σs, ci_σ²s
end



function confint(blb_ests::blbEstimates, level::Real)
    # initialize arrays for storing the CIs from each subset
    cis_βs = Array{Float64}(undef, blb_ests.all_estimates[1].p, 2, blb_ests.n_subsets) 
    cis_Σs = Array{Float64}(undef, ◺(blb_ests.all_estimates[1].q), 2, blb_ests.n_subsets) 
    cis_σ²s = Array{Float64}(undef, 1, 2, blb_ests.n_subsets) 
    @inbounds for i in 1:blb_ests.n_subsets
        cis_βs[:, :, i], cis_Σs[:, :, i], cis_σ²s[:, :, i] = confint(blb_ests.all_estimates[i], level)
    end
    ci_β  = mean(cis_βs, dims = 3)[:, :, 1]
    ci_Σ  = mean(cis_Σs, dims = 3)[:, :, 1]
    ci_σ² = mean(cis_σ²s, dims = 3)[:, :, 1]
    return ci_β, ci_Σ, ci_σ²
end
confint(blb_ests::blbEstimates) = confint(blb_ests, 0.95)


# returns fixed effect estimates
function fixef(blb_ests::blbEstimates)
    means_βs = Matrix{Float64}(undef, blb_ests.n_subsets, blb_ests.all_estimates[1].p) # n-by-p matrix 
    for i in 1:blb_ests.n_subsets
        means_βs[i, :] = mean(blb_ests.all_estimates[i].βs, dims = 1)
    end
    mean_β = mean(means_βs, dims = 1)
    return mean_β
end


# returns variance components estimates
function vc(blb_ests::blbEstimates)
    means_Σs = Array{Float64}(undef, blb_ests.all_estimates[1].q, blb_ests.all_estimates[1].q, blb_ests.n_subsets)
    # print(means_Σs, "\n")
    means_σ²s = Vector{Float64}(undef, blb_ests.n_subsets)
    for i in 1:blb_ests.n_subsets
        # print(mean(blb_ests.all_estimates[i].Σs, dims = 3), "\n")
        means_Σs[:,:,i] = mean(blb_ests.all_estimates[i].Σs, dims = 3)
        # view(means_Σs, :, :, i) .= mean(blb_ests.all_estimates[i].Σs, dims = 3)
        means_σ²s[i] = mean(blb_ests.all_estimates[i].σ²s)
    end
    mean_Σ = mean(means_Σs, dims = 3)[:, :, 1]
    mean_σ² = mean(means_σ²s)
    return mean_Σ, mean_σ²
end

function StatsBase.coeftable(blb_ests::blbEstimates, fe_ci::Matrix)
    co = fixef(blb_ests)
    # print("co = ", co, "\n")
    # pvalue = 
    # names = blb_ests.fenames

    CoefTable(
        hcat(reshape(co, :, 1), fe_ci[:, 1], fe_ci[:, 2]),
        ["Estimate", "Lower", "Upper"],
        blb_ests.fenames
        # 4 # pvalcol
    )
end


function Base.show(io::IO, blb_ests::blbEstimates)
    println("Bag of Little Boostrap (BLB) for linear mixed models.")
    println("Number of subsets: ", blb_ests.n_subsets)
    println("Number of grouping factors per subset: ", blb_ests.subset_size)
    println("Number of bootstrap samples per subset: ", blb_ests.n_boots)
    println(io)

    # calculate all CIs
    ci_β, ci_Σ, ci_σ² = confint(blb_ests)
    cnames = ["lower" "upper"]

    println("Variance Components")
    mean_Σ, mean_σ² = vc(blb_ests)
    println(io)

    println("Random Effects")
    println(io)
    println("Estimates")
    # print("blb_ests.renames = ", blb_ests.renames, "\n")
    # print("mean_Σ = ", mean_Σ, "\n")
    # show(io, [Text.(reshape(blb_ests.renames, 1, :)); mean_Σ])
    show(io, mean_Σ)
    println(io)
    println("CI")
    # show(io, [Text.(cnames); ci_Σ])
    show(io, ci_Σ)
    println(io)

    println("Residual")
    println("Estimate")
    show(io, mean_σ²)
    println(io)
    println("CI")
    # show(io, [Text.(cnames); ci_σ²])
    show(io, ci_σ²)
    println(io)
    println(io)

    
    println("Fixed-effect parameters")
    print("ci_β = ", ci_β, "\n")
    show(io, coeftable(blb_ests, ci_β))
end



