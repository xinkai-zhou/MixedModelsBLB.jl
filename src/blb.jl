

"""
    SubsetEstimates

BLB linear mixed model estimates from one subset.
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


"""
    SubsetEstimates(n_boots, p, q)

Constructor for `SubsetEstimates`.
"""
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

BLB linear mixed model estimates, which contains blb parameters and a vector of `SubsetEstimates`.
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
    runtime::Vector{Float64}
    # estimation method
    method::Symbol
end

"""
    save_bootstrap_result!(subset_estimates, β, Σ, σ²)

Save the result from one bootstrap iteration to subset_estimates

## Positional arguments 
* `subset_estimates`: an object of the SubsetEstimates type
* `i`: update the ith element of βs, Σs and σ²s
* `β`: parameter estimates for fixed effects
* `Σ`: the covariance matrix of variance components
* `σ²`: estimate of error variance 
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
    update_w!(m, w)

Update the weight vector `m.w` using `w`.
"""
function update_w!(
    m::Union{blblmmModel{T}, WSVarLmmModel{T}},
    w::Vector{Int64}
    ) where T <: BlasReal
    if typeof(m) == blblmmModel{T}
        copyto!(m.w, w)
    else
        copyto!(m.obswts, w)
    end
end

"""
    blb_one_subset(rng, m; N, subset_size, n_boots, method, solver, verbose, nonparametric_boot)

Performs Bag of Little Bootstraps on a subset. 

# Positional arguments 
- `rng`: random number generator. Default to the global rng.
- `m`: an object of type blblmmModel or WSVarLmmModel

# Keyword arguments
- `N`: number of individuals/clusters in the full dataset.
- `subset_size`: number of individuals/clusters in each subset.
- `n_boots`: number of bootstrap iterations. Default to 1000.
- `method`: a symbol, either `:ML` or `:WiSER`.
- `solver`: solver for the optimization problem. 
- `verbose`: Bool, whether to print bootstrap progress.
- `nonparametric_boot`: Bool, whether to use nonparametric bootstrap. For WiSER models, only nonparametric bootstrap is supported.

# Values
- `subset_estimates`: an object of type `SubsetEstimates`
"""
function blb_one_subset(
    rng::Random.AbstractRNG,
    m::Union{blblmmModel{T}, WSVarLmmModel{T}};
    N::Int64, 
    subset_size::Int64,
    n_boots::Int = 1000,
    method::Symbol,
    solver,
    verbose::Bool,
    nonparametric_boot::Bool
    ) where T <: BlasReal 

    if method == :ML
        MixedModelsBLB.init_ls!(m)
        MixedModelsBLB.fit!(m, solver)
    else
        WiSER.init_ls!(m)
        WiSER.fit!(m, solver, verbose = verbose)
    end

    # Initalize an instance of SubsetEstimates type for storing results
    subset_estimates = SubsetEstimates(n_boots, m.p, m.q)

    # construct the simulator
    if nonparametric_boot
        verbose && print("Using Non-parametric Bootstrap\n")
        simulator = NonparametricBootSimulator(m, N = N, subset_size = subset_size)
    else
        verbose && print("Using Parametric Bootstrap\n")
        simulator = ParametricBootSimulator(m, N = N, subset_size = subset_size)
    end

    # Bootstrapping
    @inbounds for k = 1:n_boots
        if verbose
            flush(stdout)
            print("Bootstrap iteration ", k, "\n")
        end

        if !nonparametric_boot
            # Parametric bootstrapping. Updates m.data[i].y for all i
            simulate!(rng, m, simulator)
        end

        # Get weights by drawing N i.i.d. samples from multinomial
        Random.rand!(rng, simulator.mult_dist, simulator.ns) 
        
        # Update weights in blblmmModel
        update_w!(m, simulator.ns)
        
        # Fit model on the bootstrap sample
        if method == :ML
            MixedModelsBLB.fit!(m, solver)
        else
            WiSER.fit!(m, solver, verbose = verbose)
        end
        
        # Save estimates
        if method == :ML
            save_bootstrap_result!(subset_estimates, k, m.β, m.Σ, m.σ²[1])
        else
            save_bootstrap_result!(subset_estimates, k, m.β, m.Σγ, exp(m.τ[1]))
        end
        
        if !nonparametric_boot
            # Reset model parameter to subset estimates because
            # using the bootstrap estimates from each iteration may be unstable.
            copyto!(m.β, simulator.β_subset)
            copyto!(m.Σ, simulator.Σ_subset)
            copyto!(m.σ², simulator.σ²_subset)
        end
    end
    return subset_estimates
end
blb_one_subset(m::Union{blblmmModel, WSVarLmmModel}; N::Int64, subset_size::Int64, n_boots::Int = 1000, 
                method::Symbol, solver, verbose::Bool, nonparametric_boot::Bool)  = 
    blb_one_subset(Random.GLOBAL_RNG, m; N = N, subset_size = subset_size, n_boots = n_boots, method = method, 
                    solver = solver, verbose = verbose, nonparametric_boot = nonparametric_boot)


"""
    blblmmobs(datatable)

Constructor for the type `blblmmObs`

# Positional arguments 
- `data_obs`: a table object that is compatible with Tables.jl
- `feformula`: the formula for the fixed effects
- `reformula`: the formula for the random effects
"""
function blblmmobs(data_obs, feformula::FormulaTerm, reformula::FormulaTerm)
    y, X = StatsModels.modelcols(feformula, data_obs)
    Z = StatsModels.modelmatrix(reformula, data_obs)
    return blblmmObs(y, X, Z)
end
blblmmobs(feformula::FormulaTerm, reformula::FormulaTerm) = data_obs -> blblmmobs(data_obs, feformula, reformula)

"""
    wsvarlmmobs(datatable)

Constructor for the type `WSVarLmmObs`

# Positional arguments 
- `data_obs`: a table object that is compatible with `Tables.jl`
- `feformula`: formula for the fixed effects
- `reformula`: formula for the random effects
- `wsvarformula`: formula for the fixed effects of the within-subject variance
"""
function wsvarlmmobs(data_obs, feformula::FormulaTerm, reformula::FormulaTerm, wsvarformula::FormulaTerm)
    y, X = StatsModels.modelcols(feformula, data_obs)
    Z = StatsModels.modelmatrix(reformula, data_obs)
    W = StatsModels.modelmatrix(wsvarformula, data_obs)
    return WSVarLmmObs(y, X, Z, W)
end
wsvarlmmobs(feformula::FormulaTerm, reformula::FormulaTerm, wsvarformula::FormulaTerm) = 
    data_obs -> wsvarlmmobs(data_obs, feformula, reformula, wsvarformula)

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
    subsetting!(rng, subset_id, data_columns, id_name, unique_id, cat_names, cat_levels)

Draw a subset from the full dataset. The IDs of the subset is stored in `subset_id`

# Positional arguments 
- `rng`: random number generator. Default to the global rng.
- `subset_id`: a vector for storing the IDs of the subset
- `data_columns`: an object of the AbstractColumns type, or a DataFrame
- `unique_id`: a vector of the unique ID in the full data set
- `cat_names`: a character vector of the names of the categorical variables
- `cat_levels`: a dictionary that contains the number of levels of each categorical variable
"""
function subsetting!(
    rng::Random.AbstractRNG,
    subset_id::Vector,
    data_columns,
    id_name::Symbol,
    unique_id::Vector,
    cat_names::Vector{String},
    cat_levels::Dict
    )
    if length(cat_names) == 0
        sample!(rng, unique_id, subset_id; replace = false)
    else
        # there are categorical variables
        good_subset = false
        while !good_subset
            # Sample from the full dataset
            sample!(rng, unique_id, subset_id; replace = false)
            # subset_indices = LinearIndices(id)[findall(in(blb_id_unique), id)]
            cat_levels_subset = data_columns |> 
                TableOperations.filter(x -> Tables.getcolumn(x, Symbol(id_name)) .∈ Ref(Set(subset_id))) |> 
                Tables.columns |> 
                count_levels(cat_names)
            # If the subset levels do not match the full dataset levels, 
            # skip the current iteration and take another subset
            if cat_levels_subset == cat_levels
                good_subset = true
            end
        end
    end
    sort!(subset_id)
end
subsetting!(subset_id::Vector, data_columns, id_name::Symbol, unique_id::Vector, cat_names::Vector{String}, cat_levels::Dict) = 
    subsetting!(Random.GLOBAL_RNG, subset_id, data_columns, id_name, unique_id, cat_names, cat_levels) 

"""
    x_in_y(x, sorted_y, k)

Test whether x is in a sorted vector y or not.

# Positional arguments 
- `x`: a value of type T
- `y`: a sorted vector of type T
"""    
function x_in_y(x::T, sorted_y::Vector{T}, k::Int) where T
    # test whether x is in sorted_y
    index = searchsortedfirst(sorted_y, x)
    return index ≤ k && sorted_y[index] == x
end

"""
    blb_full_data(rng, datatable; feformula, reformula, wsvarformula, id_name, cat_names, subset_size, n_subsets, n_boots, method, solver, verbose,  nonparametric_boot)

Performs Bag of Little Bootstraps on the full dataset

# Positional arguments 
- `rng`: random number generator. Default to the global rng.
- `datatable`: a data table type that is compatible with Tables.jl. 

# Keyword arguments
- `feformula`: model formula for the fixed effects.
- `reformula`: model formula for the random effects.
- `wsvarformula`: model formula for the fixed effects of the within-subject variance. For linear mixed models, it should be `@formula(y ~ 1)`. Only need to be specified when `method = :WiSER`. 
- `id_name`: name of the cluster identifier variable. String.
- `cat_names`: a vector of the names of the categorical variables.
- `subset_size`: number of clusters in the subset. 
- `n_subsets`: number of subsets.
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `method`: fitting the model by maximum-likelihood (:ML) or GEE (:WiSER)
- `verbose`: Bool, whether to print bootstrap progress (percentage completion)
- `nonparametric_boot`: Bool, whether to use Nonparametric bootstrap

# Values
- `result`: an object of the blbEstimates type
"""
function blb_full_data(
    rng::Random.AbstractRNG,
    datatable;
    feformula::FormulaTerm,
    reformula::FormulaTerm,
    wsvarformula::Union{FormulaTerm, Nothing} = nothing,
    id_name::String,
    cat_names::Vector{String} = Vector{String}(),
    subset_size::Int,
    n_subsets::Int = 10,
    n_boots::Int = 1000,
    method::Symbol = :ML,
    solver = Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm = "yes", warm_start_init_point = "yes"),
    verbose::Bool = false,
    nonparametric_boot::Bool = true
    )

    method ∉ [:ML, :WiSER] && error("`method` must be :ML or :WiSER.")
    if method == :WiSER && nonparametric_boot == false
        error("When method = :WiSER, use nonparametric bootstrap only.")
    end

    # Create Tables.Columns type for subsequent processing
    datatable_cols = Tables.columns(datatable) # This step does not allocate memory
    # Get the unique ids, which will be used for subsetting
    typeof(id_name) <: String && (id_name = Symbol(id_name))
    unique_id = unique(Tables.getcolumn(datatable_cols, id_name))
    # Same performance as: unique(Tables.columntable(TableOperations.select(datatable, id_name))[id_name])
    N = length(unique_id) # number of individuals/clusters in the full dataset
    if N < subset_size
        error(string("The subset size should not be bigger than the total number of clusters. \n", 
                        "Total number of clusters = ", N, "\n",
                        "Subset size = ", subset_size, "\n"))
    end

    # apply df-wide schema
    feformula = apply_schema(feformula, schema(feformula, datatable))
    reformula = apply_schema(reformula, schema(reformula, datatable))
    if method == :WiSER
        wsvarformula = apply_schema(wsvarformula, schema(wsvarformula, datatable))
    end

    #fenames = coefnames(feformula)[2]
    if typeof(StatsModels.coefnames(feformula)[2]) == String
        fenames = [StatsModels.coefnames(feformula)[2]]
    else
        fenames = StatsModels.coefnames(feformula)[2]
    end

    if typeof(StatsModels.coefnames(reformula)[2]) == String
        renames = [StatsModels.coefnames(reformula)[2]]
    else
        renames = StatsModels.coefnames(reformula)[2]
    end

    if method == :WiSER
        if typeof(StatsModels.coefnames(wsvarformula)[2]) == String
            wsvarnames = [StatsModels.coefnames(wsvarformula)[2]]
        else
            wsvarnames = StatsModels.coefnames(wsvarformula)[2]
        end
    end

    # By chance, some factors of a categorical variable may not show up in a subset. 
    # To make sure this does not happen, we
    # 1. count the number of levels of each categorical variable in the full data;
    # 2. for each sampled subset, we check whether the number of levels match. 
    # If they do not match, the subset is resampled.
    if length(cat_names) > 0
        cat_levels = count_levels(datatable_cols, cat_names)
    else
        cat_levels = Dict()
    end

    # Initialize an array to store the unique IDs for the subset
    subset_id = Vector{eltype(unique_id)}(undef, subset_size)
    # Initialize a vector of SubsetEstimates for storing estimates from subsets
    all_estimates = Vector{SubsetEstimates{Float64}}(undef, n_subsets)
    # Initialize a vector of obsvec
    if method == :ML
        obsvec = Vector{blblmmObs{Float64}}(undef, subset_size)
    else
        obsvec = Vector{WSVarLmmObs{Float64}}(undef, subset_size)
        clswts = ones(subset_size)
        respname = StatsModels.coefnames(feformula)[1]
    end
    # Initialize a vector for storing the runtime
    # Currently, runtime doesn't make sense when there are multiple workers
    runtime = Vector{Float64}(undef, n_subsets)
    # do grouping once, and collect
    datatable_grouped = datatable |> @groupby(_.id) |> collect

    if Distributed.nworkers() > 1
        # Using multi-processing
        futures = Vector{Future}(undef, n_subsets)
        # wks_schedule assigns each subset to a worker
        wks_schedule = Vector{Int}(undef, n_subsets)
        # nwks = Distributed.nworkers()
        wk = 2 # note that the worker number starts from 2
        wk_max = maximum(workers())
        @inbounds for i = 1:n_subsets
            wks_schedule[i] = wk
            wk == wk_max ? wk = 2 : wk += 1
        end
        flush(stdout)
        print("wks_schedule = ", wks_schedule, "\n")
        
        @inbounds for j = 1:n_subsets
            time0 = time_ns()
            # Take a subset
            subsetting!(rng, subset_id, datatable_cols, id_name, unique_id, cat_names, cat_levels)
            # Construct blblmmObs objects
            # if use_groupby
            if method == :ML
                obsvec = datatable_grouped |> @filter(x_in_y(key(_), subset_id, subset_size)) |> 
                        @map(blblmmobs(_, feformula, reformula)) |> collect |> Array{blblmmObs{Float64}, 1}
                # Construct the blblmmModel type
                m = blblmmModel(obsvec, fenames, renames, N)
            else
                obsvec = datatable_grouped |> @filter(x_in_y(key(_), subset_id, subset_size)) |> 
                        @map(wsvarlmmobs(_, feformula, reformula, wsvarformula)) |> collect |> Array{WSVarLmmObs{Float64}, 1}
                m = WSVarLmmModel(obsvec, obswts = clswts, respname = respname, meannames = fenames, 
                                    renames = renames, wsvarnames = wsvarnames, 
                                    meanformula = feformula, reformula = reformula, wsvarformula = wsvarformula, 
                                    ids = subset_id)
            end
            
            # Process this subset on worker "wks_schedule[j]"
            futures[j] = remotecall(blb_one_subset, wks_schedule[j], rng, m; 
                                    N = N, subset_size = subset_size, n_boots = n_boots, method = method, 
                                    solver = solver, verbose = verbose, nonparametric_boot = nonparametric_boot)
            # A remote call returns a Future to its result. Remote calls return immediately; 
            # the process that made the call proceeds to its next operation while the remote call happens somewhere else. 
            # You can wait for a remote call to finish by calling wait on the returned Future, 
            # and you can obtain the full value of the result using fetch.
            runtime[j] = (time_ns() - time0) / 1e9
        end
        @inbounds for j = 1:n_subsets
            # Fetch results from workers
            all_estimates[j] = fetch(futures[j])
        end
    else
        # Not using multi-processing
        @inbounds for j = 1:n_subsets
            time0 = time_ns()
            # Take a subset
            subsetting!(rng, subset_id, datatable_cols, id_name, unique_id, cat_names, cat_levels)
            # Construct blblmmObs objects
            if method == :ML
                obsvec = datatable_grouped |> @filter(x_in_y(key(_), subset_id, subset_size)) |> 
                        @map(blblmmobs(_, feformula, reformula)) |> collect |> Array{blblmmObs{Float64}, 1}
                # Construct the blblmmModel type
                m = blblmmModel(obsvec, fenames, renames, N)
            else
                obsvec = datatable_grouped |> @filter(x_in_y(key(_), subset_id, subset_size)) |> 
                        @map(wsvarlmmobs(_, feformula, reformula, wsvarformula)) |> collect |> Array{WSVarLmmObs{Float64}, 1}
                m = WSVarLmmModel(obsvec, obswts = clswts, respname = respname, meannames = fenames, 
                                    renames = renames, wsvarnames = wsvarnames, 
                                    meanformula = feformula, reformula = reformula, wsvarformula = wsvarformula, 
                                    ids = subset_id)
            end
            all_estimates[j] = blb_one_subset(rng, m; N = N, subset_size = subset_size, n_boots = n_boots, method = method, 
                                              solver = solver, verbose = verbose, nonparametric_boot = nonparametric_boot)
            runtime[j] = (time_ns() - time0) / 1e9
        end
    end
    # Create a blbEstimates instance for storing results from all subsets
    result = blbEstimates{Float64}(n_subsets, subset_size, n_boots, fenames, renames, all_estimates, runtime, method)
    return result
end

blb_full_data(datatable; feformula::FormulaTerm, reformula::FormulaTerm, wsvarformula::Union{FormulaTerm, Nothing} = nothing, id_name::String, 
                cat_names::Vector{String} = Vector{String}(), subset_size::Int, n_subsets::Int = 10, n_boots::Int = 200, 
                method::Symbol = :ML, solver = Ipopt.IpoptSolver(), verbose::Bool = false, nonparametric_boot::Bool = true) = 
    blb_full_data(Random.GLOBAL_RNG, datatable; feformula = feformula, reformula = reformula, wsvarformula = wsvarformula, 
                id_name = id_name, cat_names = cat_names, subset_size = subset_size, n_subsets = n_subsets, n_boots = n_boots, 
                method = method, solver = solver, verbose = verbose, nonparametric_boot = nonparametric_boot)


"""
    blb_db(rng, con; feformula, reformula, id_name, cat_names, subset_size, n_subsets, n_boots, solver, verbose,  nonparametric_boot)

Performs Bag of Little Bootstraps on databases.

# Positional arguments 
- `rng`: random number generator. Default to the global rng.
- `con`: an object of type `MySQL.Connection` created by the function `DBInterface.connect`.
- `table_name`: table name for the longitudinal data.

# Keyword arguments
- `feformula`: model formula for the fixed effects.
- `reformula`: model formula for the random effects.
- `id_name`: name of the cluster identifier variable. String.
- `cat_names`: a vector of the names of the categorical variables.
- `subset_size`: number of clusters in the subset. 
- `n_subsets`: number of subsets.
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `verbose`: Bool, whether to print bootstrap progress (percentage completion)
- `nonparametric_boot`: Bool, whether to use Nonparametric bootstrap

# Values
- `result`: an object of the blbEstimates type
"""
function blb_db(
    rng::Random.AbstractRNG,
    con,
    table_name::String;
    feformula::FormulaTerm,
    reformula::FormulaTerm,
    id_name::String,
    cat_names::Vector{String} = Vector{String}(),
    subset_size::Int,
    n_subsets::Int = 10,
    n_boots::Int = 1000,
    solver = Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm = "yes", warm_start_init_point = "yes"),
    verbose::Bool = false,
    # use_threads::Bool = false,
    nonparametric_boot::Bool = true
    )

    # initialize two empty vectors for storage
    fenames = String[]
    renames = String[]

    query = string("SELECT DISTINCT(", id_name, ") FROM ", table_name, ";")
    unique_id = DataFrame(DBInterface.execute(con,  query))[:, id_name]
    N = length(unique_id)
    if N < subset_size
        error(string("The subset size should not be bigger than the total number of clusters. \n", 
                        "Total number of clusters = ", N, "\n",
                        "Subset size = ", subset_size, "\n"))
    end

    # Initialize an array to store the unique IDs for the subset
    subset_id = Vector{eltype(unique_id)}(undef, subset_size)
    # Initialize a vector of SubsetEstimates for storing estimates from subsets
    all_estimates = Vector{SubsetEstimates{Float64}}(undef, n_subsets)
    # Initialize a vector of the blblmmObs objects
    obsvec = Vector{blblmmObs{Float64}}(undef, subset_size)
    # Initialize a vector for storing the runtime
    # Currently, runtime doesn't make sense when there are multiple workers
    runtime = Vector{Float64}(undef, n_subsets)

    if Distributed.nworkers() > 1
        # Using multi-processing
        futures = Vector{Future}(undef, n_subsets)
        # wks_schedule assigns each subset to a worker
        wks_schedule = Vector{Int}(undef, n_subsets)
        # nwks = Distributed.nworkers()
        wk = 2 # note that the worker number starts from 2
        wk_max = maximum(workers())
        @inbounds for i = 1:n_subsets
            wks_schedule[i] = wk
            wk == wk_max ? wk = 2 : wk += 1
        end
        flush(stdout)
        print("wks_schedule = ", wks_schedule, "\n")
        
        @inbounds for j = 1:n_subsets
            time0 = time_ns()
            
            # Take a subset
            sample!(unique_id, subset_id, replace = false, ordered = true)

            # create a table to store subset id. this makes the WHERE step below easier
            subset_table = string("subset", j)
            query = string("DROP TABLE IF EXISTS ", subset_table, ";")
            DBInterface.execute(con,  query)
            query = string("CREATE TABLE ", subset_table, " (id INT, INDEX(id));")
            DBInterface.execute(con,  query)
            for i in 1:subset_size
                query = string("INSERT INTO ", subset_table, " VALUES (", subset_id[i], ");")
                DBInterface.execute(con, query)
            end 
            # DBInterface.execute(con, "SELECT * FROM subset1") |> DataFrame
            
            # filter data using subset_id
            query = string("SELECT * FROM ", table_name, " WHERE id IN (SELECT id FROM ", subset_table, ");")
            datatable = DBInterface.execute(con,  query) |> DataFrame
            # drop the temp table
            query = string("DROP TABLE IF EXISTS ", subset_table, ";")
            DBInterface.execute(con,  query)
            
            # apply schema
            feformula = apply_schema(feformula, schema(feformula, datatable))
            reformula = apply_schema(reformula, schema(reformula, datatable))
            
            if j == 1
                # do this once on the first subset
                if typeof(coefnames(feformula)[2]) == String
                    puch!(fenames, coefnames(feformula)[2])
                else
                    temp = coefnames(feformula)[2]
                    for k in temp
                        push!(fenames, k)
                    end
                end
                if typeof(coefnames(reformula)[2]) == String
                    puch!(renames, coefnames(reformula)[2])
                else
                    temp = coefnames(reformula)[2]
                    for k in temp
                        push!(renames, k)
                    end
                end
            end

            # group by id and construct obsvec
            datatable_grouped = datatable |> @groupby(_.id) |> collect
            obsvec = datatable_grouped|> @map(MixedModelsBLB.blblmmobs(_, feformula, reformula)) |> collect |> 
                        Array{blblmmObs{Float64}, 1}
            
            # Construct the blblmmModel type
            m = blblmmModel(obsvec, fenames, renames, N)

            # Process this subset on worker "wks_schedule[j]"
            futures[j] = remotecall(blb_one_subset, wks_schedule[j], rng, m; 
                                    n_boots = n_boots, solver = solver, verbose = verbose, 
                                    nonparametric_boot = nonparametric_boot)
            # A remote call returns a Future to its result. Remote calls return immediately; 
            # the process that made the call proceeds to its next operation while the remote call happens somewhere else. 
            # You can wait for a remote call to finish by calling wait on the returned Future, 
            # and you can obtain the full value of the result using fetch.
            runtime[j] = (time_ns() - time0) / 1e9
        end
        @inbounds for j = 1:n_subsets
            # Fetch results from workers
            all_estimates[j] = fetch(futures[j])
        end
    else
        # Not using multi-processing
        @inbounds for j = 1:n_subsets
            time0 = time_ns()

            # Take a subset
            sample!(unique_id, subset_id, replace = false, ordered = true)

            # create a table to store subset id. this makes the WHERE step below easier
            subset_table = string("subset", j)
            query = string("DROP TABLE IF EXISTS ", subset_table, ";")
            DBInterface.execute(con,  query)
            query = string("CREATE TABLE ", subset_table, " (id INT, INDEX(id));")
            DBInterface.execute(con,  query)
            for i in 1:subset_size
                query = string("INSERT INTO ", subset_table, " VALUES (", subset_id[i], ");")
                DBInterface.execute(con, query)
            end 
            
            # filter data using subset_id
            query = string("SELECT * FROM ", table_name, " WHERE id IN (SELECT id FROM ", subset_table, ");")
            datatable = DBInterface.execute(con,  query) |> DataFrame
            # drop the temp table
            query = string("DROP TABLE IF EXISTS ", subset_table, ";")
            DBInterface.execute(con,  query)

            # apply schema
            feformula = apply_schema(feformula, schema(feformula, datatable))
            reformula = apply_schema(reformula, schema(reformula, datatable))
            
            if j == 1
                # do this once on the first subset
                if typeof(coefnames(feformula)[2]) == String
                    puch!(fenames, coefnames(feformula)[2])
                else
                    temp = coefnames(feformula)[2]
                    for k in temp
                        push!(fenames, k)
                    end
                end
                if typeof(coefnames(reformula)[2]) == String
                    puch!(renames, coefnames(reformula)[2])
                else
                    temp = coefnames(reformula)[2]
                    for k in temp
                        push!(renames, k)
                    end
                end
            end   

            # group by id and construct obsvec
            datatable_grouped = datatable |> @groupby(_.id) |> collect
            obsvec = datatable_grouped|> @map(MixedModelsBLB.blblmmobs(_, feformula, reformula)) |> collect |> 
                        Array{blblmmObs{Float64}, 1}
            
            # Construct the blblmmModel type
            m = blblmmModel(obsvec, fenames, renames, N)
            all_estimates[j] = blb_one_subset(rng, m; n_boots = n_boots, solver = solver, verbose = verbose, nonparametric_boot = nonparametric_boot)
            runtime[j] = (time_ns() - time0) / 1e9
        end
    end

    # Create a blbEstimates instance for storing results from all subsets
    result = blbEstimates{Float64}(n_subsets, subset_size, n_boots, fenames, renames, all_estimates, runtime)
    return result
end





"""
    confint(subset_ests, level)

Calculate confidence intervals using estimates from one subset.

# Positional arguments 
- `subset_ests`: an object of type `SubsetEstimates`
- `level`: confidence level, usually set to 0.95
"""
function confint(subset_ests::SubsetEstimates, level::Real)
    # ci_βs = Matrix{Float64}(undef, subset_ests.p, 2) # p-by-2 matrix
    # for i in 1:subset_ests.p
    #     ci_βs[i, :] = StatsBase.percentile(view(subset_ests.βs, :, i), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
    # end
    # q = subset_ests.q
    # ci_Σs = Matrix{Float64}(undef, ◺(q), 2)
    # k = 1
    # # For Σ, we get the CI for the diagonals first, then the upper off-diagonals
    # @inbounds for i in 1:q
    #     ci_Σs[k, :] = StatsBase.percentile(view(subset_ests.Σs, i, i, :), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
    #     k += 1
    # end
    # @inbounds for i in 1:q, j in (i+1):q
    #     ci_Σs[k, :] = StatsBase.percentile(view(subset_ests.Σs, i, j, :), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
    #     k += 1
    # end
    # ci_σ²s = reshape(StatsBase.percentile(subset_ests.σ²s, 100 * [(1 - level) / 2, 1 - (1-level) / 2]), 1, 2)
    # return ci_βs, ci_Σs, ci_σ²s
    ci_βs = Matrix{Float64}(undef, subset_ests.p, 2) # p-by-2 matrix
    for i in 1:subset_ests.p
        if all(y -> isnan(y), view(subset_ests.βs, :, i))
            ci_βs[i, :] .= NaN
        else
            ci_βs[i, :] = StatsBase.percentile(filter(y -> !isnan(y), view(subset_ests.βs, :, i)), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
        end
    end
    q = subset_ests.q
    ci_Σs = Matrix{Float64}(undef, ◺(q), 2)
    k = 1
    # For Σ, we get the CI for the diagonals first, then the upper off-diagonals
    @inbounds for i in 1:q
        if all(y -> isnan(y), view(subset_ests.Σs, i, i, :))
            ci_Σs[k, :] .= NaN
        else
            ci_Σs[k, :] = StatsBase.percentile(filter(y -> !isnan(y), view(subset_ests.Σs, i, i, :)), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
        end
        k += 1
    end
    @inbounds for i in 1:q, j in (i+1):q
        if all(y -> isnan(y), view(subset_ests.Σs, i, j, :))
            ci_Σs[k, :] .= NaN
        else
            ci_Σs[k, :] = StatsBase.percentile(filter(y -> !isnan(y), view(subset_ests.Σs, i, j, :)), 100 * [(1 - level) / 2, 1 - (1-level) / 2])
        end
        k += 1
    end
    if all(y -> isnan(y), subset_ests.σ²s)
        ci_σ²s = [NaN, NaN]
    else
        ci_σ²s = reshape(StatsBase.percentile(filter(y -> !isnan(y), subset_ests.σ²s), 100 * [(1 - level) / 2, 1 - (1-level) / 2]), 1, 2)
    end
    return ci_βs, ci_Σs, ci_σ²s
end


"""
    confint(blb_ests, level)

Calculate confidence intervals using estimates from all subsets.

# Positional arguments 
- `blb_ests`: an object of type `blbEstimates`
- `level`: confidence level, usually set to 0.95
"""
function confint(blb_ests::blbEstimates, level::Real)
    # initialize arrays for storing the CIs from each subset
    cis_βs = Array{Float64}(undef, blb_ests.all_estimates[1].p, 2, blb_ests.n_subsets) 
    cis_Σs = Array{Float64}(undef, ◺(blb_ests.all_estimates[1].q), 2, blb_ests.n_subsets) 
    cis_σ²s = Array{Float64}(undef, 1, 2, blb_ests.n_subsets) 
    @inbounds for i in 1:blb_ests.n_subsets
        cis_βs[:, :, i], cis_Σs[:, :, i], cis_σ²s[:, :, i] = confint(blb_ests.all_estimates[i], level)
    end

    if any(y -> isnan(y), cis_βs)
        ci_β = fill(NaN, (blb_ests.all_estimates[1].p, 2))
    else
        ci_β  = mean(cis_βs, dims = 3)[:, :, 1]
    end

    if any(y -> isnan(y), cis_Σs)
        ci_Σ = fill(NaN, (◺(blb_ests.all_estimates[1].q), 2))
    else
        ci_Σ  = mean(cis_Σs, dims = 3)[:, :, 1]
    end

    if any(y -> isnan(y), cis_σ²s)
        ci_σ² = fill(NaN, 1, 2)
    else
        ci_σ² = mean(cis_σ²s, dims = 3)[:, :, 1]
    end

    return ci_β, ci_Σ, ci_σ²
end
confint(blb_ests::blbEstimates) = confint(blb_ests, 0.95)




"""
    fixef(blb_ests)

Calculate BLB fixed effect estimates, which are averages of fixed effect estimates from all subsets

# Positional arguments 
- `blb_ests`: an object of type `blbEstimates`
"""
# returns fixed effect estimates
function fixef(blb_ests::blbEstimates)
    means_βs = Matrix{Float64}(undef, blb_ests.n_subsets, blb_ests.all_estimates[1].p) # n-by-p matrix 
    for i in 1:blb_ests.n_subsets
        means_βs[i, :] = mean(blb_ests.all_estimates[i].βs, dims = 1)
    end
    mean_β = mean(means_βs, dims = 1)
    return mean_β
end


"""
    vc(blb_ests)

Calculate BLB variance components estimates, which are averages of variance component estimates from all subsets

# Positional arguments 
- `blb_ests`: an object of type `blbEstimates`
"""
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



function StatsBase.coeftable(ests::Vector, ci::Matrix, varnames::Vector{String})
    CoefTable(
        hcat(reshape(ests, :, 1), ci[:, 1], ci[:, 2]),
        ["Estimate", "CI Lower", "CI Upper"],
        varnames
        # 4 # pvalcol
    )
end

function vectorize_Σ(Σ::Matrix)
    q  = size(Σ, 1)
    v = Vector{Float64}(undef, ◺(q))
    idx = 1
    # Extract the diagonals first
    @inbounds for i in 1:q
        v[idx] = Σ[i, i]
        idx += 1
    end
    # Extract the upper off-diagonals
    @inbounds for i in 1:q, j in (i+1):q
        v[idx] = Σ[i, j]
        idx += 1
    end
    return v
end

function vc_names(blb_ests::blbEstimates)
    q = length(blb_ests.renames)
    re_names = Vector{String}(undef, ◺(q))
    idx = 1
    # Extract the diagonals' names first
    @inbounds for i in 1:q
        re_names[idx] = blb_ests.renames[i]
        idx += 1
    end
    # Extract the upper off-diagonals' names
    @inbounds for i in 1:q, j in (i+1):q
        re_names[idx] = string(blb_ests.renames[i], " : ", blb_ests.renames[j])
        idx += 1
    end
    return vcat(re_names, "Residual")
end

function Base.show(io::IO, blb_ests::blbEstimates)
    println("Bag of Little Boostrap (BLB) for linear mixed models.")
    println("Method: ", blb_ests.method)
    println("Number of subsets: ", blb_ests.n_subsets)
    println("Number of grouping factors per subset: ", blb_ests.subset_size)
    println("Number of bootstrap samples per subset: ", blb_ests.n_boots)
    println("Confidence interval level: 95%")
    println(io)

    # calculate all CIs and mean estimates
    ci_β, ci_Σ, ci_σ² = confint(blb_ests)
    mean_β = fixef(blb_ests)
    mean_Σ, mean_σ² = vc(blb_ests)

    println("Variance Components parameters")
    show(io, StatsBase.coeftable(vcat(vectorize_Σ(mean_Σ), mean_σ²), vcat(ci_Σ, ci_σ²), vc_names(blb_ests)))
    println(io)
    println(io)
    
    println("Fixed-effect parameters")
    show(io, StatsBase.coeftable(vec(mean_β), ci_β, blb_ests.fenames))
end


