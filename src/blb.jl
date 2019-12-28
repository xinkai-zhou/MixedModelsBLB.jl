

"""
    blb_one_subset(y, X, Z, id; n_boots, solver, verbose)

Performs Bag of Little Bootstraps on a subset. 

# Positional arguments 
- `y`:
- `X`:
- `Z`:
- `id`: 

# Keyword arguments
- `n_boots`:  
- `solver`
- `verbose`

# Values
- `β̂`: 
- `Σ̂`: 
- `τ̂`: 
"""
function blb_one_subset(
    # positional arguments
    y::Vector{T}, 
    X::Matrix{T}, 
    Z::Matrix{T}, 
    id::Vector{Int64};
    # keyword arguments
    n_boots::Int64 = 1000,
    solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    verbose::Bool = false
    ) where T <: BlasReal 

    b, p, q = length(unique(id)), size(X, 2), size(Z, 2)
    
    # We may have to store these results as matrices because in blb_full_data()
    # we initialize  β̂ as Vector{Matrix{Float64}}(undef, n_subsets) 

    # Initialize arrays for storing the results
    β̂ = [Vector{Float64}(undef, p) for i = 1:n_boots]
    # If we use fill(Vector{Float64}(undef, p), s*r), then all elements of this vector
    # refer to the same empty vector. So if we use copyto!() to update, all elements of 
    # the vector will be updated with the same value.
    Σ̂ = [Matrix{Float64}(undef, q, q) for i = 1:n_boots]
    τ̂ = zeros(0)

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

    # Initalize parameters
    init_β!(m) 
    m.Σ .= Diagonal(ones(size(obs[1].Z, 2))) # initialize Σ with identity

    # Fit LMM using the subsample and get parameter estimates
    fit!(m) 
    copyto!(β_b, m.β)
    copyto!(Σ_b, m.Σ)
    copyto!(τ_b, m.τ)
    
    # Bootstrapping
    for k = 1:n_boots
        verbose && print("Bootstrap iteration", k, "\n")
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

        # Get weights by drawing N i.i.d. samples from multinomial
        rand!(Multinomial(N, ones(b)/b), ns) 
        
        # Update weights in blblmmModel
        update_w!(m, ns)
        
        # Use weighted loglikelihood to fit the bootstrapped dataset
        fit!(m)
        
        # extract estimates
        # i = (j-1) * r + k # the index for storage purpose
        copyto!(β̂[k], m.β)
        copyto!(Σ̂[k], m.Σ) # copyto!(Σ̂[i], m.Σ) doesn't work. how to index into a vector of matrices?
        push!(τ̂, m.τ[1]) # ?? better ways to do this?
        
    end
    return β̂, Σ̂, τ̂
end



"""
    blb_full_data(y, X, Z, id; n_subsets, r, solver, verbose)

Performs Bag of Little Bootstraps on the full dataset.

# Positional arguments 
- `y`:
- `X`:
- `Z`:
- `id`: 

# Keyword arguments
- `b`:
- `n_subsets`:
- `subset_size`:  
- `solver`:
- `verbose`:

# Values
- `β̂`: A s * subset_size vector of vectors
- `Σ̂`: 
- `τ̂`: 
"""

function blb_full_data(
    # positional arguments
    y::Vector{T}, 
    X::Matrix{T}, 
    Z::Matrix{T}, 
    id::Vector{T};
    # keyword arguments
    subset_size::Int64 = floor(sqrt(length(unique(id)))),
    n_subsets::Int64 = 10,
    n_boots::Int64 = 1000,
    solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    verbose::Bool = false
    ) where T <: BlasReal 

    p, q = size(X, 2), size(Z, 2)
    # Initialize arrays for storing the results
    # !! maybe use three dimensional arrays to avoid ugly subsetting    
    β̂ = [Vector{Float64}(undef, p) for i = 1:(n_subsets * subset_size)]
    Σ̂ = [Matrix{Float64}(undef, q, q) for i = 1:(n_subsets * subset_size)]
    τ̂ = zeros(0)

    blb_id = fill(0, subset_size)
    # multi-threading? julia 1.3, add a macro?
    for j = 1:n_subsets
        sample!(id, blb_id; replace = false)
        sort!(blb_id)
        β̂[((j-1) * n_boots + 1):(j * n_boots)], 
        Σ̂[((j-1) * n_boots + 1):(j * n_boots)], 
        τ̂[((j-1) * n_boots + 1):(j * n_boots)] = 
        blb_one_subset(
            y[id .== blb_id],
            X[id .== blb_id],
            Z[id .== blb_id],
            id[id .== blb_id];
            n_boots = n_boots,
            solver = solver,
            verbose = verbose
        )
    end
    return β̂, Σ̂, τ̂
end




"""
    blb_full_data(file; n_subsets, r, solver, verbose)

Performs Bag of Little Bootstraps on the full dataset.

# Positional arguments 
- `file`: File path.

# Keyword arguments
- `subset_size`: Size of the BLB subset. 
- `n_subsets`: Number of subsets, default to 10
- `subset_size`: Number of Monte Carlo iterations, default to 1000
- `solver`: 
- `verbose`: 

# Values
- `β̂`: A n_subsets * subset_size vector of vectors
- `Σ̂`: 
- `τ̂`: 
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
    id_name::Vector{Int32},
    cat_names::Vector{String},
    subset_size::Int64,
    n_subsets::Int64 = 10,
    n_boots::Int64 = 1000,
    solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    verbose::Bool = false
    ) where T <: BlasReal 

    # Get variable names from the formula
    lhs_name = [string(x) for x in StatsModels.termvars(f.lhs)]
    rhs_name = [string(x) for x in StatsModels.termvars(f.rhs)]
    var_name = Tuple(x for x in vcat(lhs_name, rhs_name)) # var names need to be in tuples for using select()
    
    # Connect to the dataset/folder
    ftable = JuliaDB.loadtable(
        file, 
        datacols = filter(x -> x != nothing, var_name)
    )

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
    # original initialization: [Vector{Float64}(undef, p) for i = 1:(n_subsets * subset_size)]
    Σ̂ = Vector{Matrix{Float64}}(undef, n_subsets) #[Matrix{Float64}(undef, q, q) for i = 1:(n_subsets * subset_size)]
    τ̂ = zeros(0)

    
    # Load the id column
    id = JuliaDB.select(ftable, Symbol(id_name))
    # Since the data may not be balanced, we find the number of repeats per subject
    id_counts = StatsBase.countmap(id)
    id_unique = unique(id)

    # Initialize an array to store the unique blb IDs
    blb_id_unique = fill(0, subset_size)

    Threads.@threads  for j = 1:n_subsets
        # https://julialang.org/blog/2019/07/multithreading

        # Count the total number of observations in the subset.
        # n_obs = 0
        # for key in blb_id_unique
        #     n_obs += id_counts[key]
        # end
        # intercept = fill(1., n_obs)

        subset_good = false
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
        # Create the LinearMixedModel object using the subset
        # !!! using @views
        m = LinearMixedModel(f, ftable[subset_indices, ])
        # Extract y, X, Z, id
        β̂[j] = # return from blb_one_subset(), a matrix
        β̂[((j-1) * n_boots + 1):(j * n_boots)], 
        Σ̂[((j-1) * n_boots + 1):(j * n_boots)], 
        τ̂[((j-1) * n_boots + 1):(j * n_boots)] = blb_one_subset(
            m.y, m.X, transpose(first(m.reterms).z), id[subset_indices]; 
            n_boots = n_boots, solver = solver, verbose = verbose
        )
        ####
        # y = select(ftable, Symbol(y_name))[subset_indices, ]
        # # if fe is missing, X is just intercept, otherwise 
        # isnothing(fe_name) ? X = intercept : X = hcat(intercept, JuliaDB.select(ftable, map(Symbol, fe_name))[subset_indices, ])
        # isnothing(re_name) ? Z = intercept : Z = hcat(intercept, JuliaDB.select(ftable, map(Symbol, re_name))[subset_indices, ])
        ####
        j += 1
    end
    return β̂, Σ̂, τ̂
end




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