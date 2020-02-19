

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
- `τ̂`: a vector of size n_boots
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
    # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000),
    MoM_init = false,
    verbose::Bool = false
    ) where T <: BlasReal 

    # writedlm("subset-X.csv", X, ',')
    # writedlm("subset-Z.csv", Z, ',')
    # writedlm("subset-id.csv", id, ',')

    b, p, q = length(unique(id)), size(X, 2), size(Z, 2)
    
    # We may have to store these results as matrices because in blb_full_data()
    # we initialize  β̂ as Vector{Matrix{Float64}}(undef, n_subsets) 

    # Initialize arrays for storing the results
    β̂ = Matrix{Float64}(undef, n_boots, p)
    # print("initialized β̂ =", β̂, "\n")
    # print("size of β̂ =", size(β̂), "\n")
    Σ̂ = Matrix{Float64}(undef, n_boots, q) # only save the diagonals
    τ̂ = zeros(0)

    # β̂ = [Vector{Float64}(undef, p) for i = 1:n_boots]
    # # If we use fill(Vector{Float64}(undef, p), s*r), then all elements of this vector
    # # refer to the same empty vector. So if we use copyto!() to update, all elements of 
    # # the vector will be updated with the same value.
    # Σ̂ = [Matrix{Float64}(undef, q, q) for i = 1:n_boots]
    # τ̂ = zeros(0)

    # Initialize an array for storing multinomial counts
    ns = zeros(b)

    # Initialize a vector of the blblmmObs objects
    obs = Vector{blblmmObs{Float64}}(undef, b)
    for (i, grp) in enumerate(unique(id))
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
    τ_b = similar(m.τ)
    
    # # Testing MoM initialization
    # print("Initialize with MoM and print results\n")
    # init_MoM!(m)
    # print("MoM m.β = ", m.β, "\n")
    # print("MoM m.τ[1] = ", m.τ[1], "\n")
    # print("MoM m.Σ = ", m.Σ, "\n\n\n")

    # MixedModels.fit!(lmm)
    # copyto!(m.β, lmm.β)
    # m.τ[1] = 1 / (lmm.σ^2)
    # m.Σ .= Diagonal([i^2 for i in lmm.sigmas[1]]) 
    # print("Initialize with LMM and print results\n")
    # print("LMM m.β = ", m.β, "\n")
    # print("LMM m.τ[1] = ", m.τ[1], "\n")
    # print("LMM m.Σ = ", m.Σ, "\n\n\n")

    # Initalize parameters
    if MoM_init 
        # Method of Moments initialization
        init_MoM!(m)
        # This updates β, τ and Σ
    else
        # use MixedModels.jl to initialize
        # fit model
        MixedModels.fit!(lmm)
        # print(lmm)
        # print(lmm.β)
        # print(propertynames(lmm))
        # print(lmm.sigmas)
        # copy value over
        copyto!(m.β, lmm.β)
        # copyto!(m.τ, [1 / (lmm.σ^2)])
        m.τ[1] = 1 / (lmm.σ^2)
        m.Σ .= Diagonal([i^2 for i in lmm.sigmas[1]]) # initialize Σ
    end
    # init_β!(m) # the original LS initialization
    
    print("After initilization,\n")
    print("m.β = ", m.β, "\n")
    print("m.τ[1] = ", m.τ[1], "\n")
    print("m.Σ = ", m.Σ, "\n")
    
    # Fit LMM using the subsample and get parameter estimates
    fit!(m; solver = solver) 
    # will remove this later because this should be exactly the same as MixedModels.fit!
    copyto!(β_b, m.β)
    copyto!(Σ_b, m.Σ)
    copyto!(τ_b, m.τ)
    print("finished fitting on the subset \n")
    # print("β_b = ", β_b, "\n")
    # print("Σ_b = ", Σ_b, "\n")

    print("After fitting on the subset,\n")
    print("m.β = ", m.β, "\n")
    print("m.τ[1] = ", m.τ[1], "\n")
    print("m.Σ = ", m.Σ, "\n")

    # Bootstrapping
    for k = 1:n_boots
        verbose && print("Bootstrap iteration ", k, "\n\n\n\n\n\n")
        # Generate a parametric bootstrap sample of y and update m.y 
        # in place by looping over blblmmModel.
        for bidx = 1:b
            copyto!(
                m.data[bidx].y, 
                m.data[bidx].X * β_b + # fixed effect
                m.data[bidx].Z * rand(MvNormal(zeros(size(Σ_b, 1)), Σ_b)) + # random effect
                rand(Normal(0, sqrt(1 / τ_b[1])), length(m.data[bidx].y)) # error, standard normal
            )
        end
        # # save the data
        # y = Vector{Float64}()
        # for bidx = 1:b
        #     append!(y, m.data[bidx].y)
        # end
        # writedlm("bootstrap-y.csv", y, ',')

        # Get weights by drawing N i.i.d. samples from multinomial
        rand!(Multinomial(N, ones(b)/b), ns) 
        # writedlm("bootstrap-ns.csv", ns, ',')

        # Update weights in blblmmModel
        update_w!(m, ns)
        
        print("Inside bootstrap, before fitting \n")
        print("m.β = ", m.β, "\n")
        print("m.τ[1] = ", m.τ[1], "\n")
        print("m.Σ = ", m.Σ, "\n")

        # Use weighted loglikelihood to fit the bootstrapped dataset
        fit!(m; solver = solver)
        
        print("Inside bootstrap, after fitting\n")
        print("m.β = ", m.β, "\n")
        print("m.τ[1] = ", m.τ[1], "\n")
        print("m.Σ = ", m.Σ, "\n")

        # print("m.β = ", m.β, "\n")
        # print("diag(m.Σ) = ", diag(m.Σ), "\n")

        # extract estimates
        β̂[k, :] .= m.β 
        # if the assignment is for certain rows of a matrix, then ".=" works fine and we don't need copyto()
        # actually copyto() wouldn't work.
        Σ̂[k, :] .= diag(m.Σ)
        # copyto!(β̂[k, :], m.β)
        # copyto!(Σ̂[k, :], diag(m.Σ))
        push!(τ̂, m.τ[1])

        # reset model parameter to subset estimates because 
        # using the bootstrap estimates from each iteration may be unstable.
        copyto!(m.β, β_b)
        copyto!(m.Σ, Σ_b)
        copyto!(m.τ, τ_b)
        # m.τ[1] = τ_b[1]

        # print("β̂[k, :] = ", β̂[k, :], "\n")
        # print("Σ̂[k, :] = ", Σ̂[k, :], "\n")
        # Original implementation
        # i = (j-1) * r + k # the index for storage purpose
        # copyto!(β̂[k], m.β)
        # copyto!(Σ̂[k], m.Σ) # copyto!(Σ̂[i], m.Σ) doesn't work. how to index into a vector of matrices?
        # push!(τ̂, m.τ[1]) # ?? better ways to do this?
    end
    # print("updated β̂ =", β̂, "\n")
    # print("updated Σ̂ =", Σ̂, "\n")
    return β̂, Σ̂, τ̂
end



"""
    blb_full_data(y, X, Z, id, N; subset_size, n_subsets, n_boots, solver, verbose)

Performs Bag of Little Bootstraps on the full dataset. This interface is intended for smaller datasets that can be loaded in memory.


# Positional arguments 
- `y`: response vector
- `X`: design matrix for fixed effects
- `Z`: design matrix for random effects
- `id`: cluster identifier

# Keyword arguments
- `subset_size`: number of clusters in the subset. Default to the square root of the total number of clusters.
- `n_subsets`: number of subsets.
- `n_boots`: number of bootstrap iterations. Default to 1000
- `solver`: solver for the optimization problem. 
- `verbose`: print extra information ???

# Values
- `β̂`: a vector (of size n_subsets) of matrices (of size n_boots-by-p)
- `Σ̂`: a vector (of size n_subsets) of matrices (of size n_boots-by-q, only saves the diagonals of Σ̂)
- `τ̂`: a vector (of size n_subsets) of vectors (of size n_boots)
"""

function blb_full_data(
    # positional arguments
    df::DataFrame,
    f::FormulaTerm;
    # y::Vector{T}, 
    # X::Matrix{T}, 
    # Z::Matrix{T}, 
    # id::Vector{Int64};
    # keyword arguments
    subset_size::Int64 = floor(sqrt(length(unique(id)))),
    n_subsets::Int64 = 10,
    n_boots::Int64 = 1000,
    # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000),
    verbose::Bool = false
    ) where T <: BlasReal 

    p, q = size(X, 2), size(Z, 2)
    N = length(unique(id))
    # print("p = ", p, "\n")
    # Initialize arrays for storing the results
    # !! maybe use three dimensional arrays to avoid ugly subsetting    
    β̂ = [Vector{Float64}(undef, p) for i = 1:(n_subsets * subset_size)]
    Σ̂ = [Matrix{Float64}(undef, q, q) for i = 1:(n_subsets * subset_size)]
    τ̂ = zeros(0)

    blb_id = fill(0, subset_size)
    
    Threads.@threads  for j = 1:n_subsets
        sample!(id, blb_id; replace = false)
        sort!(blb_id)

        df_subset = df[subset_indices, :]
        categorical!(df, Symbol("id"))
        m = LinearMixedModel(f, df)

        β̂[((j-1) * n_boots + 1):(j * n_boots)], 
        Σ̂[((j-1) * n_boots + 1):(j * n_boots)], 
        τ̂[((j-1) * n_boots + 1):(j * n_boots)] = 
        blb_one_subset(
            y[id .== blb_id],
            X[id .== blb_id],
            Z[id .== blb_id],
            id[id .== blb_id],
            N = N;
            n_boots = n_boots,
            solver = solver,
            verbose = verbose
        )
    end
    return β̂, Σ̂, τ̂
end



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
- `τ̂`: a vector (of size n_subsets) of vectors (of size n_boots)
"""
function blb_full_data(
    # positional arguments
    # !!! could be a folder
    file::String,
    f::FormulaTerm;
    # keyword arguments
    # will remove these names later
    # y_name::String,
    # fe_name::Union{String, Vector{String}, nothing} = nothing,
    # re_name::Union{String, Vector{String}, nothing} = nothing,
    # this is the only thing we need for subsampling
    id_name::String,
    cat_names::Vector{String},
    subset_size::Int64,
    n_subsets::Int64 = 10,
    n_boots::Int64 = 1000,
    MoM_init = false,
    # solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000),
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

    # df = DataFrame(ftable)
    # categorical!(df, Symbol("id"))
    # print("first(df, 5) = ", first(df, 5), "\n")
    # print("formula = ", f, "\n")
    # # categorical!(df, Symbol.(cat_names))
    # lmm = LinearMixedModel(f, df)
    # print(propertynames(lmm, true), "\n")
    # print(fieldnames(typeof(lmm)), "\n")

    # By chance, certain factors may not show up in a subset. To make sure this does not happen, we need to 
    # count the number of levels of the categorical variables
    cat_levels = Dict{String, Int32}()
    for cat_name in cat_names
        cat_levels[cat_name] = length(unique(JuliaDB.select(ftable, Symbol(cat_name))))
    end

    # The size of the parameter vectors can only be decided after creating the LinearMixedModel type, 
    # because that's when we do recoding of categorical vars.

    # Initialize arrays for storing the results
    β̂ = Vector{Matrix{Float64}}(undef, n_subsets)
    # print("blb_full_data initialized β̂ = ", β̂, "\n") 
    # original initialization: [Vector{Float64}(undef, p) for i = 1:(n_subsets * subset_size)]
    Σ̂ = Vector{Matrix{Float64}}(undef, n_subsets) 
    #[Matrix{Float64}(undef, q, q) for i = 1:(n_subsets * subset_size)]
    τ̂ = Vector{Vector{Float64}}(undef, n_subsets) 
    
    # Load the id column
    id = JuliaDB.select(ftable, Symbol(id_name))
    # Since the data may not be balanced, we find the number of repeats per subject
    # id_counts = StatsBase.countmap(id)
    id_unique = unique(id)
    N = length(id_unique)

    # Initialize an array to store the unique IDs for the subset
    blb_id_unique = fill(0, subset_size)

    Threads.@threads for j = 1:n_subsets
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
        end
        # print("subset_indices = ", subset_indices, "\n")
        # Create the LinearMixedModel object using the subset
        # !!! using @views
        # m = LinearMixedModel(f, ftable[subset_indices, ])

        # To use LinearMixedModel(), 
        # we need to transform the id column in ftable to categorical type.
        # Currently I do this by converting the table to DataFrame.
        # In later releases of MixedModels.jl, they will no longer require "id"
        # to be cateogorical. I will change the script then.
        df = DataFrame(ftable[subset_indices, ])
        categorical!(df, Symbol("id"))
        # print("first(df, 5) = ", first(df, 5), "\n")
        # print("formula = ", f, "\n")
        # categorical!(df, Symbol.(cat_names))
        lmm = LinearMixedModel(f, df)
        # print(propertynames(lmm, true), "\n")
        # print(fieldnames(typeof(lmm)), "\n")

        # print("m.X", m.X, "\n")
        # return from blb_one_subset(), a matrix
        β̂[j], Σ̂[j], τ̂[j] = blb_one_subset(
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
    end
    # print("blb_full_data β̂ = ", β̂, "\n") 
    return β̂, Σ̂, τ̂
end


# function testlmm(
#     file::String,
#     f::FormulaTerm,
#     id_name::String
# )
#     # lhs_name = [string(x) for x in StatsModels.termvars(f.lhs)]
#     # rhs_name = [string(x) for x in StatsModels.termvars(f.rhs)]
#     # var_name = Tuple(x for x in vcat(lhs_name, rhs_name)) # var names need to be in tuples for using select()
    
    
#     # ftable = JuliaDB.loadtable(
#     #     file, 
#     #     datacols = filter(x -> x != nothing, vcat(lhs_name, rhs_name))
#     # )
#     df = CSV.read(file)
#     df = DataFrame(df)
#     print(first(df, 5), "\n")
#     print("formula = ", f, "\n")
#     categorical!(df, Symbol(id_name))
#     print(first(df, 5), "\n")

#     lmm = LinearMixedModel(f, df)
#     @edit propertynames(lmm, true)
    
#     MixedModels.fit!(lmm)
    
#     print(lmm, "\n")
    
#     # print(MixedModels.σs(lmm), "\n")
    
#     print(propertynames(lmm, true), "\n")
# end

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





# The following function tries to use formula interface.

# """
#     blb_full_data(formula interface, data; s, r, solver, verbose)

# Performs Bag of Little Bootstraps on the full dataset using formula interface.

# # Positional arguments 
# - `mformula::FormulaTerm`:
# - `data`:

# # Keyword arguments
# - `subset_size`:
# - `s`:
# - `n_boots`:  
# - `solver`:
# - `verbose`:

# # Values
# - `β̂`: A s * r vector of vectors
# - `Σ̂`: 
# - `τ̂`: 
# """

# function blb_full_data(
#     # positional arguments
#     f::FormulaTerm, 
#     data::Matrix{T};
#     # keyword arguments
#     b::Int64,
#     s::Int64,
#     r::Int64,
#     solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
#     verbose::Bool = false
#     ) where T <: BlasReal 

#     # Initialize arrays for storing the results
#     β̂ = [Vector{Float64}(undef, p) for i = 1:(s * r)]
#     Σ̂ = [Matrix{Float64}(undef, q, q) for i = 1:(s * r)]
#     τ̂ = zeros(0)

#     # Construct y, X, Z, id from the formula
#     tbl, _ = StatsModels.missing_omit(tbl, f)
#     form = apply_schema(f, schema(f, tbl, contrasts), LinearMixedModel)
#     # tbl, _ = StatsModels.missing_omit(tbl, form)

#     y, Xs = modelcols(form, tbl)

#     y = reshape(float(y), (:, 1)) # y as a floating-point matrix
#     T = eltype(y)

#     y = 
#     X = 
#     Z = 
#     id =
#     β̂, Σ̂, τ̂ = blb_full_data(
#         y, X, Z, id; 
#         b = b, s = s, r = r, verbose = verbose
#     )
#     return β̂, Σ̂, τ̂
# end



# The following function tries to call functions from MixedModel.jl to 
# construct the design matrices for both fixed and random effects.
# function print_matrices(
#     f::FormulaTerm, 
#     tbl::DataFrame
#     ) 

#     # The following functions are inter-connected, for example:
#     # (1) modelcols even uses the function remat(); (2) apply_schema() uses the 
#     # MixedModel type. So if I borrow his approach, then I basically need 
#     # to borrow almost the entire package. 

#     # Construct y, X, Z, id from the formula
#     tbl, _ = StatsModels.missing_omit(tbl, f)
#     form = apply_schema(f, schema(f, tbl, contrasts), LinearMixedModel)
#     # tbl, _ = StatsModels.missing_omit(tbl, form)

#     y, Xs = modelcols(form, tbl)

#     y = reshape(float(y), (:, 1)) # y as a floating-point matrix
#     T = eltype(y)

#     return y, X, Z, id
# end