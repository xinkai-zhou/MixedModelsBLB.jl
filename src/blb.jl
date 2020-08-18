

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
    runtime::Vector{Float64}
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
- `rng`: random number generator. Default to the global rng.
- `m`: an object of the blblmmModel type

# Keyword arguments
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `verbose`: Bool, whether to print bootstrap progress

# Values
- `subset_estimates`: an object of the SubsetEstimates type
"""
function blb_one_subset(
    rng::Random.AbstractRNG,
    m::blblmmModel{T};
    n_boots::Int = 1000,
    solver = Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm = "yes", warm_start_init_point = "yes"),
    verbose::Bool = false
    ) where T <: BlasReal 

    # Initalize model parameters
    init_ls!(m, verbose)
    
    # Fit LMM on the subset
    fit!(m, solver)

    # Initalize an instance of SubsetEstimates type for storing results
    subset_estimates = SubsetEstimates(n_boots, m.p, m.q)

    # construct the simulator type
    simulator = Simulator(m)

    # Bootstrapping
    @inbounds for k = 1:n_boots
        verbose && print("Bootstrap iteration ", k, "\n")

        # Parametric bootstrapping. Updates m.data[i].y for all i
        simulate!(rng, m, simulator)

        # Get weights by drawing N i.i.d. samples from multinomial
        rand!(simulator.mult_dist, simulator.ns) 
        
        # Update weights in blblmmModel
        update_w!(m, simulator.ns)
        
        # Fit model on the bootstrap sample
        fit!(m, solver)
        
        # Save estimates
        save_bootstrap_result!(subset_estimates, k, m.β, m.Σ, m.σ²[1])

        # Reset model parameter to subset estimates because
        # using the bootstrap estimates from each iteration may be unstable.
        copyto!(m.β, simulator.β_subset)
        copyto!(m.Σ, simulator.Σ_subset)
        copyto!(m.σ², simulator.σ²_subset)
    end
    return subset_estimates
end
blb_one_subset(m::blblmmModel; n_boots::Int = 1000, solver, verbose::Bool = false) = 
    blb_one_subset(Random.GLOBAL_RNG, m; n_boots = n_boots, solver = solver, verbose = verbose)

# function replicate(f::Function, n::Integer; use_threads=false)
#     if use_threads
#         # no macro version yet: https://github.com/timholy/ProgressMeter.jl/issues/143
#         # get the type
#         rr = f()
#         # pre-allocate
#         results = [rr for _ in Base.OneTo(n)]
#         Threads.@threads for idx = 2:n
#             results[idx] = f()
#         end
#     else
#         results = [f() for _ in Base.OneTo(n)]
#     end
#     results
# end

# """
#     blb_one_subset(m; n_boots, solver, verbose)

# Performs Bag of Little Bootstraps on a subset. 

# # Positional arguments 
# - `m`: an object of the blblmmModel type

# # Keyword arguments
# - `n_boots`: number of bootstrap iterations. Default to 1000
# - `solver`: solver for the optimization problem. 
# - `verbose`: Bool, whether to print bootstrap progress (percentage completion)

# # Values
# - `subset_estimates`: an object of the SubsetEstimates type
# """
# function blb_one_subset_2(
#     rng::Random.AbstractRNG,
#     m::blblmmModel{T};
#     n_boots::Int64 = 1000,
#     solver = Ipopt.IpoptSolver(print_level = 0),
#     verbose::Bool = false,
#     use_threads::Bool = false
#     ) where T <: Real 

#     print("m.β = ", m.β, "\n")
#     print("m.data[1].y = ", m.data[1].y, "\n")

#     βsc, Σsc, σ²sc, p, q, mc = similar(m.β), similar(m.Σ), similar(m.σ²), m.p, m.q, deepcopy(m)

#     init_ls!(mc, verbose)
#     print("mc.β = ", mc.β, "\n")
#     print("mc.data[1].y = ", mc.data[1].y, "\n")
#     simulatorc = Simulator(mc)

#     mc_threads = [m]
#     simulatorc_threads = [simulatorc]
#     βsc_threads = [βsc]
#     Σsc_threads = [Σsc]
#     σ²sc_threads = [σ²sc]

#     if use_threads
#         Threads.resize_nthreads!(mc_threads)
#         Threads.resize_nthreads!(βsc_threads)
#         Threads.resize_nthreads!(Σsc_threads)
#         Threads.resize_nthreads!(σ²sc_threads)
#     end

#     rnglock = ReentrantLock()
#     samp = replicate(n_boots, use_threads = use_threads) do
#         mod = mc_threads[Threads.threadid()]
#         init_ls!(mod, verbose)
#         print("mod.β = ", mod.β, "\n")
#         print("mod.data[1].y = ", mod.data[1].y, "\n")
#         simulator = simulatorc_threads[Threads.threadid()]
#         local βsc = βsc_threads[Threads.threadid()]
#         local Σsc = Σsc_threads[Threads.threadid()]
#         local σ²sc = σ²sc_threads[Threads.threadid()]
#         lock(rnglock)
#         simulate!(rng, mod, simulator)
#         rand!(simulator.mult_dist, simulator.ns) 
#         update_w!(mod, simulator.ns)
#         unlock(rnglock)
#         fit!(mod, solver)
#         (
#             #  β = copyto!(βsc, mod.β),
#             #  Σ = copyto!(Σsc, mod.Σ),
#             #  σ² = copyto!(σ²sc, mod.σ²)
#          β = copyto!(βsc, mod.β),
#          Σ = copyto!(Σsc, mod.Σ),
#          σ² = copyto!(σ²sc, mod.σ²)
#         )
#     end

#     return samp
# end


"""
    blblmmobs(datatable)

Construct the blblmmObs type

# Positional arguments 
- `data_obs`: a table object that is compatible with Tables.jl
- `feformula`: the formula for the fixed effects
- `reformula`: the formula for the fixed effects
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
end
subsetting!(subset_id::Vector, data_columns, id_name::Symbol, unique_id::Vector, cat_names::Vector{String}, cat_levels::Dict) = 
    subsetting!(Random.GLOBAL_RNG, subset_id, data_columns, id_name, unique_id, cat_names, cat_levels) 

"""
    x_in_y!(x, y, k)

Test whether x is in y or not.

# Positional arguments 
- `x`: a value of type T
- `y`: a vector of type T
"""    
function x_in_y(x::T, y::Vector{T}, k::Int) where T
    # test whether x is in subset_id
    result = searchsortedfirst(y, x) <= k
end


"""
    blb_full_data(rng, datatable; feformula, reformula, id_name, cat_names, subset_size, n_subsets, n_boots, solver, verbose, use_threads)

Performs Bag of Little Bootstraps on the full dataset

# Positional arguments 
- `rng`: random number generator. Default to the global rng.
- `datatable`: a data table type that is compatible with Tables.jl. 

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
- `use_threads`: Bool, whether to use multithreading. Default to false.

# Values
- `result`: an object of the blbEstimates type
"""
function blb_full_data(
    rng::Random.AbstractRNG,
    datatable;
    feformula::FormulaTerm,
    reformula::FormulaTerm,
    id_name::String,
    cat_names::Vector{String} = Vector{String}(),
    subset_size::Int,
    n_subsets::Int = 10,
    n_boots::Int = 1000,
    solver = Ipopt.IpoptSolver(),
    verbose::Bool = false,
    use_threads::Bool = false,
    newway::Bool = false
    )
    # !!!!!!!!!!!!!! WE SHOULD BE ABLE TO AVOID CREATING datatable_cols
    # Create Tables.Columns type for subsequent processing
    datatable_cols = Tables.columns(datatable)
    # Get the unique ids, which will be used for subsetting
    typeof(id_name) <: String && (id_name = Symbol(id_name))
    unique_id = unique(Tables.getcolumn(datatable_cols, id_name))
    N = length(unique_id) # number of individuals/clusters in the full dataset
    if N < subset_size
        error(string("The subset size should be no bigger than the total number of clusters. \n", 
                        "Total number of clusters = ", N, "\n",
                        "Subset size = ", subset_size, "\n"))
    end

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
    if length(cat_names) > 0
        cat_levels = count_levels(datatable_cols, cat_names)
    else
        cat_levels = Dict()
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
    # new way of constructing obsvec
    if newway
        # do grouping once, and collect
        datatable_grouped = datatable |> @groupby(_.id) |> collect
    end

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
        
        @inbounds for j = 1:n_subsets
            time0 = time_ns()
            # Take a subset
            subsetting!(subset_id, datatable_cols, id_name, unique_id, cat_names, cat_levels)
            # Construct blblmmObs objects
            if newway
                obsvec = datatable_grouped |> @filter(x_in_y(key(_), subset_id, subset_size)) |> 
                @map(blblmmobs(_, feformula, reformula)) |> collect |> Array{blblmmObs{Float64}, 1}
            else
                Threads.@threads for i in 1:subset_size
                    obsvec[i] = datatable_cols |> 
                        TableOperations.filter(x -> Tables.getcolumn(x, id_name) == subset_id[i]) |> 
                        Tables.columns |> 
                        blblmmobs(feformula, reformula)
                end
            end
            # Construct the blblmmModel type
            m = blblmmModel(obsvec, fenames, renames, N, use_threads) 
            # Process this subset on worker "wks_schedule[j]"
            futures[j] = remotecall(blb_one_subset, wks_schedule[j], rng, m; n_boots = n_boots, solver = solver, verbose = verbose)
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
            subsetting!(subset_id, datatable_cols, id_name, unique_id, cat_names, cat_levels)
            # Construct blblmmObs objects
            if newway
                obsvec = datatable_grouped |> @filter(x_in_y(key(_), subset_id, subset_size)) |> 
                @map(blblmmobs(_, feformula, reformula)) |> collect |> Array{blblmmObs{Float64}, 1}
            else
                Threads.@threads for i in 1:subset_size
                    obsvec[i] = datatable_cols |> 
                        TableOperations.filter(x -> Tables.getcolumn(x, id_name) == subset_id[i]) |> 
                        Tables.columns |> 
                        blblmmobs(feformula, reformula)
                end
            end
            # Construct the blblmmModel type
            m = blblmmModel(obsvec, fenames, renames, N, use_threads) 
            all_estimates[j] = blb_one_subset(
                rng,
                m;
                n_boots = n_boots, 
                solver = solver, 
                verbose = verbose
            )
            runtime[j] = (time_ns() - time0) / 1e9
        end
    end

    # Create a blbEstimates instance for storing results from all subsets
    result = blbEstimates{Float64}(n_subsets, subset_size, n_boots, fenames, renames, all_estimates, runtime)
    return result
end

blb_full_data(datatable; feformula::FormulaTerm, reformula::FormulaTerm, id_name::String, 
                cat_names::Vector{String} = Vector{String}(), subset_size::Int, n_subsets::Int = 10, n_boots::Int = 200, 
                solver = Ipopt.IpoptSolver(), verbose::Bool = false, use_threads::Bool = false, newway::Bool = false) = 
    blb_full_data(Random.GLOBAL_RNG, datatable; feformula = feformula, reformula = reformula, id_name = id_name, 
                    cat_names = cat_names, subset_size = subset_size, n_subsets = n_subsets, n_boots = n_boots, 
                    solver = solver, verbose = verbose, use_threads = use_threads, newway = newway)


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

# Make the output nicer
# function vctable(blb_ests::blbEstimates)
#     mean_Σ, mean_σ² = vc(blb_ests)

# end

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
    show(io, coeftable(blb_ests, ci_β))
end


