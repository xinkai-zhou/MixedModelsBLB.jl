
    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    σ²inv = 1 / σ²[1]

    ###########
    # objective
    ###########
    copyto!(obs.storage_qq_1, obs.ztz) 
    # storage_qq_1 = L'Z'Z
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
    needgrad && copyto!(obs.∇L, obs.storage_qq_1)
    # storage_qq_1 = L'Z'Z L
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
    # storage_qq_2 = L'Z'Z L
    needgrad && copyto!(obs.storage_qq_2, obs.storage_qq_1)
    # form σ²I + L'Z'Z L
    @inbounds for i in 1:q
        obs.storage_qq_1[i, i] += σ²[1]
    end
    # storage_qq_1 = upper cholesky factor of σ²I + L'Z'Z L
    LAPACK.potrf!('U', obs.storage_qq_1) 
    # ∇L = chol^{-1} L'Z'Z
    needgrad && BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq_1, obs.∇L)
    if needhess
        # storage_qp = L'Z'X
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, copyto!(obs.storage_qp, obs.ztx))
        # storage_qp = chol^{-1} L'Z'X
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq_1, obs.storage_qp)
    end
    # calculate rtr as yty - β'xty - β'xtr (reason: we will need xtr in ∇β)
    # first calculate xtr
    BLAS.gemv!('N', T(-1), obs.xtx, β, T(1), copyto!(obs.xtr, obs.xty))
    rtr = obs.yty - dot(β, obs.xty) - dot(β, obs.xtr)
    # ztr = Z'r = -Z'Xβ + Z'y
    BLAS.gemv!('N', T(-1), obs.ztx, β, T(1), copyto!(obs.ztr, obs.zty))
    # storage_q_1 = L'Z'r
    BLAS.trmv!('L', 'T', 'N', ΣL, copyto!(obs.storage_q_1, obs.ztr))
    # storage_q_1 = chol^{-1} L'Z'r
    BLAS.trsv!('U', 'T', 'N', obs.storage_qq_1, obs.storage_q_1)
    # calculate the loglikelihood
    logl = n * log(2π) + (n-q) * log(σ²[1])
    @inbounds for i in 1:q
        logl += 2 * log(obs.storage_qq_1[i, i])

        # # the diag of chol may be <=0 due to numerical reasons. 
        # # if this happens, set logl to be -Inf.
        # if obs.storage_qq_1[i, i] <= 0
        #     logl = -Inf
        #     return logl
        # else 
        #     logl += 2 * log(obs.storage_qq_1[i, i])    
        # end
    end
    # the quadratic form will be used in grad too
    qf = dot(obs.storage_q_1, obs.storage_q_1)
    logl += σ²inv * (rtr - qf)
    logl /= -2
    
    ###########
    # gradient
    ###########
    if needgrad
        # caluclate storage_qq_1 = (σ²I + L'Z'Z L)^{-1} 
        LAPACK.potri!('U', obs.storage_qq_1)
        LinearAlgebra.copytri!(obs.storage_qq_1, 'U') 
        # calculate tr_minvltztzl, which will be used in ∇σ². storage_qq_2 = L'Z'Z L
        tr_minvltztzl = dot(obs.storage_qq_1, obs.storage_qq_2)
        # calculate storage_qq_1 = LMinvL', which will be used repeatedly
        BLAS.trmm!('L', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
        BLAS.trmm!('R', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
        
        # ∇β
        # then calculate storage_q_1 = LM^{-1}L'Z'r
        BLAS.gemv!('N', T(1), obs.storage_qq_1, obs.ztr, T(0), obs.storage_q_1)
        # calculate ∇β
        BLAS.gemv!('T', T(-1), obs.ztx, obs.storage_q_1, T(1), copyto!(obs.∇β, obs.xtr))
        obs.∇β .*= σ²inv

        # ∇σ²
        obs.∇σ²[1] = -σ²inv * (n - tr_minvltztzl)
        # storage_q_2 = Z'ZLM^{-1}L'Z'r
        BLAS.gemv!('N', T(1), obs.ztz, obs.storage_q_1, T(0), obs.storage_q_2)
        obs.∇σ²[1] += abs2(σ²inv) * (rtr - 2 * qf + dot(obs.storage_q_1, obs.storage_q_2))
        obs.∇σ²[1] /= 2

        # ∇Σ (get ∇Σ on the obs level, then get ∇L on the model level)
        # currently ∇L = chol^{-1} L'Z'Z
        # calculate storage_qq_2 = Z'ZLMinvL'Z'Z using a rank-k update
        BLAS.syrk!('U', 'T', T(1), obs.∇L, T(0), obs.storage_qq_2)
        LinearAlgebra.copytri!(obs.storage_qq_2, 'U')
        if needhess
            # Hσ²L
            # currently storage_qq_1 = LMinvL', storage_qq_2 = Z'ZLMinvL'Z'Z
            tr_lminvltztzlminvltztz = dot(obs.storage_qq_1, obs.storage_qq_2)
            # storage_qq_3 = LMinvL'Z'Z
            mul!(obs.storage_qq_3, obs.storage_qq_1, obs.ztz)
            # storage_qq_1 = Z'ZLMinvL'Z'Z LMinvL'Z'Z
            mul!(obs.storage_qq_1, obs.storage_qq_2, obs.storage_qq_3)
            BLAS.axpy!(T(1), obs.ztz, obs.storage_qq_1)
            BLAS.axpy!(T(-2), obs.storage_qq_2, obs.storage_qq_1)
            lmul!(abs2(σ²inv), obs.storage_qq_1)
            # now storage_qq_1 = Z'Ω^{-2}Z, right multiply L to get Z'Ω^{-2}ZL
            BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
            # print("storage_qq_1 = ", obs.storage_qq_1, "\n")
            # D = [1 0 0; 0 1 0; 0 0 0; 0 0 1] 
            # obs.Hσ²L .= vec(vec(obs.storage_qq_1)' * D)
            vech!(obs.Hσ²L, obs.storage_qq_1)
        end
        # calculate storage_qq_2 = -(Z'Z - Z'ZLMinvL'Z'Z)
        BLAS.axpy!(T(-1), obs.ztz, obs.storage_qq_2)
        # storage_qq_2 = -Z'ΩinvZ = -σ²inv * (Z'Z - Z'ZLMinvL'Z'Z)
        lmul!(σ²inv, obs.storage_qq_2)
        copyto!(obs.∇L, obs.storage_qq_2)
        if needhess
            # HLL
            fill!(obs.HLL, T(0))
            # let storage_qq_2 = Z'ΩinvZ
            lmul!(T(-1), obs.storage_qq_2)
            LinearAlgebra.copytri!(obs.storage_qq_2, 'U')
            copyto!(obs.storage_qq_1, obs.storage_qq_2)
            # let storage_qq_1 = Z'ΩinvZL
            BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1)
            Ct_At_kron_A_KC!(obs.HLL, obs.storage_qq_1)
            # storage_qq_1 = L'Z'ΩinvZL, storage_qq_2 = Z'ΩinvZ
            BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
            Ct_A_kron_B_C!(obs.HLL, obs.storage_qq_1, obs.storage_qq_2)
        end
        # currently storage_q_2 = Z'ZLM^{-1}L'Z'r
        # calculate storage_q_2 = Z'ZLM^{-1}L'Z'r - Z'r 
        BLAS.axpy!(T(-1), obs.ztr, obs.storage_q_2)
        # ∇L = -Z'ΩinvZ + Z'Ωinvrr'ΩinvZ
        BLAS.syr!('U', abs2(σ²inv), obs.storage_q_2, obs.∇L)
        # update ∇L
        LinearAlgebra.copytri!(obs.∇L, 'U')
    end

    ###########
    # hessian
    ###########
    if needhess
        # Hββ
        BLAS.syrk!('U', 'T', T(-1), obs.storage_qp, T(1), copyto!(obs.Hββ, obs.xtx))
        lmul!(σ²inv, obs.Hββ)

        # Hσ²σ²
        obs.Hσ²σ²[1] = (abs2(σ²inv) * (n - 2 * tr_minvltztzl + tr_lminvltztzlminvltztz)) / 2

        # Hσ²L
        # done above

        # HLL
        # done above
    end

    logl








# using Tables, TableOperations, CSV, DataFrames, StatsBase
# function count_levels(data_columns::Union{Tables.AbstractColumns, DataFrames.DataFrame}, cat_names::Vector{String})
#     cat_levels = Dict{String, Int64}()
#     @inbounds for cat_name in cat_names
#         cat_levels[cat_name] = length(countmap(Tables.getcolumn(data_columns, Symbol(cat_name))))
#     end
#     return cat_levels
# end
# count_levels(cat_names::Vector{String}) = data_columns -> count_levels(data_columns, cat_names)


# data_columns = CSV.read("data/minimal-test.csv")
# id_name = "A"
# subset_id = [0,0,0,0]
# unique_id = [1,2,3,4]
# cat_names = ["B"]
# dc = Tables.columns(data_columns)
# cat_levels = count_levels(dc, cat_names)
# # data_columns |> 
# # TableOperations.filter(x -> Tables.getcolumn(x, Symbol(id_name)) .∈ Ref(Set(subset_id))) |> 
# # Tables.columns  |> 
# # count_levels(cat_names)

# function subset1!(
#     subset_id::Vector{Int64},
#     data_columns::DataFrame,
#     id_name::String,
#     unique_id::Vector{Int64},
#     cat_names::Vector{String},
#     cat_levels::Dict{String, Int64}
#     )
#     good_subset = false
#     while !good_subset
#         # Sample from the full dataset
#         sample!(unique_id, subset_id; replace = false)
#         # subset_indices = LinearIndices(id)[findall(in(blb_id_unique), id)]
#         if length(cat_names) > 0
#             cat_levels_subset = data_columns |> 
#                 TableOperations.filter(x -> Tables.getcolumn(x, Symbol(id_name)) .∈ Ref(Set(subset_id))) |> 
#                 Tables.columns |> 
#                 count_levels(cat_names)
#             # If the subset levels do not match the full dataset levels, 
#             # skip the current iteration and take another subset
#             if cat_levels_subset == cat_levels
#                 good_subset = true
#             end
#         else 
#             good_subset = true
#         end
#     end
# end
# subset1!(subset_id, dc, id_name, unique_id, cat_names, cat_levels)

# function subsetting!(
#     subset_id::Vector{Int}, 
#     # data_columns::Union{Tables.AbstractColumns, DataFrames.DataFrame};
#     data_columns::DataFrame;
#     id_name::String,
#     unique_id::Vector{Int}, 
#     cat_names::Vector{String},
#     cat_levels::Dict{String, Int}
#     )
#     good_subset = false
#     while !good_subset
#         # Sample from the full dataset
#         sample!(unique_id, subset_id; replace = false)
#         # subset_indices = LinearIndices(id)[findall(in(blb_id_unique), id)]
#         if length(cat_names) > 0
#             cat_levels_subset = data_columns |> 
#                 TableOperations.filter(x -> Tables.getcolumn(x, Symbol(id_name)) .∈ Ref(Set(subset_id))) |> 
#                 Tables.columns |> 
#                 count_levels(cat_names)
#             # If the subset levels do not match the full dataset levels, 
#             # skip the current iteration and take another subset
#             if cat_levels_subset == cat_levels
#                 subset_good = true
#             end
#         else 
#             subset_good = true
#         end
#     end
# end
# subsetting!(subset_id, dc, id_name, unique_id, cat_names, cat_levels)


# using DataFrames, Tables
# function foo(cols::Tables.Columns, x::Symbol)
#     print(Tables.getcolumn(cols, x))
# end
# dat = DataFrame(x = ones(3), y = zeros(3))
# dc = Tables.columns(dat)
# typeof(dc)

# function foo(cols, x::Symbol)
#     print(Tables.getcolumn(cols, x))
# end

# """
#     extract_Σ!(Σ, lmm::LinearMixedModel)

# Extract the variance-covariance matrix of variance components 
# from a LinearMixedModel object for initialization.
# """
# function extract_Σ!(Σ, lmm::LinearMixedModel)
#     σρ = MixedModels.VarCorr(lmm).σρ
#     q = size(Σ, 1) #length(σρ[1][1])
#     @inbounds @views for i in 1:q
#         Σ[i, i] = (σρ[1][1][i])^2
#         @inbounds for j in (i+1):q
#             Σ[i, j] = σρ[1][2][(j-1)] * σρ[1][1][i] * σρ[1][1][j]
#         end
#     end
#     LinearAlgebra.copytri!(Σ, 'U')
#     return(Σ)
# end


"""
blb_one_subset(lmm, y, X, Z, id, N; n_boots, solver, LS_init, verbose)

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
- `σ̂²`: a vector of size n_boots
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
LS_init = false,
verbose::Bool = false
) where T <: BlasReal 

b, p, q = length(Set(id)), size(X, 2), size(Z, 2)

# move model construction to blb_full_data
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

# Initalize parameters
init_ls!(m)
# if LS_init 
#     # LS initialization
#     init_ls!(m) # This updates β, σ² and Σ
# else
#     # use MixedModels.jl to initialize
#     MixedModels.fit!(lmm)
#     copyto!(m.β, lmm.β)
#     m.σ²[1] = lmm.σ^2
#     extract_Σ!(m.Σ, lmm)
# end

# Fit LMM using the subsample and get parameter estimates
fit!(m; solver = solver) 
# # Initialize arrays for storing subset estimates
# β_b = similar(m.β)
# Σ_b = similar(m.Σ)
# σ²_b = similar(m.σ²)
# # Save subset estimates for parametric bootstrapping
# copyto!(β_b, m.β)
# copyto!(Σ_b, m.Σ)
# copyto!(σ²_b, m.σ²)

# print("β_b = ", β_b, "\n")
# print("Σ_b = ", Σ_b, "\n")
# print("σ²_b = ", σ²_b, "\n")
# Initalize an instance of SubsetEstimates type for storing results
subset_estimates = SubsetEstimates(n_boots, m.p, m.q)

# construct the simulator type
simulator = Simulator(m, b, N)

# Bootstrapping
@inbounds for k = 1:n_boots
    verbose && print("Bootstrap iteration ", k, "\n")

    # Parametric bootstrapping
    simulate!(m, simulator, b)

    # Get weights by drawing N i.i.d. samples from multinomial
    rand!(simulator.mult_dist, simulator.ns) 
    # print("simulator.ns[1:10] = ", simulator.ns[1:10], "\n")
    # Update weights in blblmmModel
    update_w!(m, simulator.ns)
    
    # Fit model on the bootstrap sample
    fit!(m; solver = solver)
    # print("m.β = ", m.β, "\n")
    # print("m.Σ = ", m.Σ, "\n")
    # save estimates
    save_bootstrap_result!(subset_estimates, k, m.β, m.Σ, m.σ²[1])

    # Reset model parameter to subset estimates because 
    # using the bootstrap estimates from each iteration may be unstable.
    copyto!(m.β, simulator.β_b)
    copyto!(m.Σ, simulator.Σ_b)
    copyto!(m.σ², simulator.σ²_b)
end
return subset_estimates
end


"""
blb_full_data(file, f; id_name, cat_names, subset_size, n_subsets, n_boots, LS_init, solver, verbose)

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
- `σ̂²`: a vector (of size n_subsets) of vectors (of size n_boots)
"""
function blb_full_data(
# positional arguments
file::String,
f::FormulaTerm;
# dtable;
# meanformula::FormulaTerm,
# reformula::FormulaTerm,
# keyword arguments
id_name::String,
cat_names::Vector{String},
subset_size::Int64,
n_subsets::Int64 = 10,
n_boots::Int64 = 1000,
LS_init = true,
solver = Ipopt.IpoptSolver(),
verbose::Bool = false
)

# # Get variable names from the formula
# meanformula_lhs = [string(x) for x in StatsModels.termvars(meanformula.lhs)]
# meanformula_rhs = [string(x) for x in StatsModels.termvars(meanformula.rhs)]
# reformula_lhs = [string(x) for x in StatsModels.termvars(reformula.lhs)]
# reformula_rhs = [string(x) for x in StatsModels.termvars(reformula.rhs)]
# # var names need to be in tuples for using select()
# var_name = Tuple(x for x in unique(vcat(meanformula_lhs, meanformula_rhs, reformula_lhs, reformula_rhs))) 

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

# Load the id column
id = JuliaDB.select(ftable, Symbol(id_name))
id_unique = unique(id)
N = length(id_unique) # number of clusters in the full dataset
# Initialize an array to store the unique IDs for the subset
blb_id_unique = fill(0, subset_size)

# Initialize a vector of SubsetEstimates
all_estimates = Vector{SubsetEstimates{Float64}}(undef, n_subsets)

# Threads.@threads for j = 1:n_subsets
@inbounds for j = 1:n_subsets
    # https://julialang.org/blog/2019/07/multithreading

    # !!!! Put in into a function
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

    @views df = DataFrame(ftable[subset_indices, ])
    categorical!(df, Symbol("id"))
    lmm = LinearMixedModel(f, df)

    all_estimates[j] = blb_one_subset(
        lmm,
        lmm.y, 
        lmm.X, 
        copy(transpose(first(lmm.reterms).z)), 
        id[subset_indices],
        N; 
        n_boots = n_boots, 
        LS_init = LS_init,
        solver = solver, 
        verbose = verbose
    )
end

# Create a blbEstimates instance for storing results from all subsets
result = blbEstimates{Float64}(n_subsets, n_boots, all_estimates)
return result
end


# Bad Woodbury for objective for gradient
    ###########
    # objective
    ###########
    # update_res!(obs, β)
    # # Here we compute the inverse of Σ. To avoid allocation, first copy Σ to storage_qq,
    # # then calculate the in-place cholesky, then in-place inverse (only the upper-tri in touched)
    # copyto!(obs.storage_qq, Σ)
    # # print("before potrf, obs.storage_qq = ", obs.storage_qq, "\n")
    # LAPACK.potrf!('U', obs.storage_qq) # in-place cholesky
    # # Calculate the logdet(Σ) part of logl
    # logl = 0
    # # print("after potrf, obs.storage_qq = ", obs.storage_qq, "\n")
    # @inbounds for i in 1:q
    #     if obs.storage_qq[i, i] <= 0
    #         logl = -Inf
    #         return logl
    #     else 
    #         # (-1//2) logdet(Σ) = (-1//2) \Sum 2*log(obs.storage_qq[i, i])
    #         #                   = - \Sum log(obs.storage_qq[i, i])
    #         logl -= log(obs.storage_qq[i, i])    
    #     end
    # end
    # LAPACK.potri!('U', obs.storage_qq) # in-place inverse (only the upper-tri in touched.)
    # # obs.storage_qq = Σ^{-1} + τZ'Z
    # BLAS.axpy!(τ[1], obs.ztz, obs.storage_qq)
    # # BLAS.syrk!('U', 'T',  τ[1], obs.Z, T(1), obs.storage_qq) # only the upper-tri is touched
    # LinearAlgebra.copytri!(obs.storage_qq, 'U')
    # storage_qq_chol = cholesky!(obs.storage_qq, Val(true); check = false) 
    # if rank(storage_qq_chol) < q # Since storage_qq_chol is of Cholesky type, rank doesn't call SVD
    #     logl = -Inf # set logl to -Inf and return
    #     return logl
    # end
    # # (Σ^{-1} + τZ'Z)^{-1} Z'
    # ldiv!(obs.storage_qn, storage_qq_chol, transpose(obs.Z))
    # # -τ^2 * Z (Σ^{-1} + τZ'Z)^{-1} Z'
    # BLAS.gemm!('N', 'N', -τ[1]^2, obs.Z, obs.storage_qn, T(0), obs.storage_nn)
    # @inbounds for i in 1:n
    #     obs.storage_nn[i, i] += τ[1]
    # end
    # # Now obs.storage_nn is Ω^{-1}.
    # BLAS.gemv!('T', T(1), obs.storage_nn, obs.res, 0., obs.storage_n1)
    # # Currently logl equals logdet(Σ)
    # logl += (-1//2) * (logdet(storage_qq_chol) + n * log(1/τ[1]) + dot(obs.res, obs.storage_n1))
    # # print("New way logl = ", logl, "\n")
    
    # ###########
    # # gradient
    # ###########
    # if needgrad
    #     # wrt β
    #     # copyto!(obs.∇β, vec(BLAS.gemm('T', 'N', obs.X, obs.storage_n1)))
    #     BLAS.gemv!('T', T(1), obs.X, obs.storage_n1, T(0), obs.∇β)
    #     # wrt L, New code using Woodbury
    #     # \Omegabf^{-1}\Zbf_i
    #     BLAS.gemm!('N', 'N', T(1), obs.storage_nn, obs.Z, T(0), obs.storage_nq)
    #     #  -\Zbf_i' \Omegabf^{-1}\Zbf_i 
    #     BLAS.gemm!('T', 'N', T(-1), obs.Z, obs.storage_nq, T(0), obs.∇L)
    #     # \Zbf_i'\Omegabf^{-1}\rbf 
    #     BLAS.gemv!('T', T(1), obs.storage_nq, obs.res, T(0), obs.storage_1q)
    #     # -\Zbf_i' \Omegabf^{-1}\Zbf_i + \Zbf_i'\Omegabf^{-1}\rbf \rbf'\Omegabf^{-1}\Zbf_i
    #     BLAS.ger!(1., obs.storage_1q, obs.storage_1q, obs.∇L)
    #     # new code with Woodbury
    #     obs.∇τ[1] = (1/(2 * τ[1]^2)) * (tr(obs.storage_nn) - dot(obs.storage_n1, obs.storage_n1))
    # end


# # Old objective and gradient code
# # old way of calculating the objective 
# update_res!(obs, β)
# mul!(obs.storage_qn, Σ, transpose(obs.Z))
# mul!(obs.V, obs.Z, obs.storage_qn)
# # V = obs.Z * Σ * obs.Z' + (1 / τ) * I
# # calculate once 
# τ_inv = (1 / τ[1])
# @inbounds for i in 1:n
#     obs.V[i, i] += τ_inv 
# end
# # Using the cholesky appraoch
# Vchol = cholesky!(Symmetric(obs.V), Val(true); check = false) 
# if rank(Vchol) < n # Since Vchol is of Cholesky type, rank(Vchol) doesn't call SVD
#     logl = -Inf # set logl to -Inf and return
#     return logl
# end
# ldiv!(obs.storage_n1, Vchol, obs.res)
# logl = (-1//2) * (logdet(Vchol) + dot(obs.res, obs.storage_n1))
# print("Old way logl = ", logl, "\n")

# if needgrad
#     # wrt β
#     # copyto!(obs.∇β, vec(BLAS.gemm('T', 'N', obs.X, obs.storage_n1)))
#     BLAS.gemv!('T', T(1), obs.X, obs.storage_n1, T(0), obs.∇β)

#     # old code for gradient wrt L
#     ldiv!(obs.storage_nq, Vchol, obs.Z)
#     BLAS.gemm!('T', 'N', -1., obs.Z, obs.storage_nq, 0., obs.∇L)
#     BLAS.gemv!('T', 1., obs.storage_nq, obs.res, 0., obs.storage_1q)
#     BLAS.ger!(1., obs.storage_1q, obs.storage_1q, obs.∇L)
#     # !!! use trmm for triangular matrix multiplication
#     # Since ΣL is the same for all clusters, instead of doing rmul! repeatedly, 
#     # we will do it once in the aggregate step.
#     # rmul!(obs.∇L, ΣL)
    
#     # # old old code for gradient wrt L
#     # ldiv!(obs.storage_nq, Vchol, obs.Z)
#     # # Original code ----
#     # # copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.Z, obs.storage_nq))
#     # # # BLAS.gemm!('T', 'N', obs.Z, obs.storage_nq, false, obs.storage_qq)
#     # # rmul!(obs.storage_qq, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
#     # # obs.∇L .= - obs.storage_qq
#     # # New code ----
#     # BLAS.gemm!('T', 'N', -1., obs.Z, obs.storage_nq, false, obs.∇L)
#     # rmul!(obs.∇L, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
#     # # copyto!(obs.storage_1q, BLAS.gemm('T', 'N', reshape(obs.res, (n, 1)), obs.storage_nq))
#     # BLAS.gemv!('T', 1., obs.storage_nq, obs.res, false, obs.storage_1q)
#     # # copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.storage_1q, obs.storage_1q))
#     # # copyto!(obs.storage_qq, BLAS.gemm('T', 'N', reshape(obs.storage_1q, (1, q)), reshape(obs.storage_1q, (1, q))))
#     # # Since we initialized storage_qq as 0, the following should work
#     # obs.storage_qq .= 0.
#     # BLAS.ger!(1., obs.storage_1q, obs.storage_1q, obs.storage_qq)
#     # rmul!(obs.storage_qq, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
#     # obs.∇L .+= obs.storage_qq 

#     # new code with Woodbury
#     # obs.∇τ[1] = (1/(2 * τ[1]^2)) * (tr(obs.storage_nn) - dot(obs.storage_n1, obs.storage_n1))

#     # # old code for gradient wrt τ
#     # # Since Vchol and V are no longer needed, we can calculate in-place inverse of obs.V
#     # # obs.V holds the in place cholesky of V
#     LAPACK.potri!('U', obs.V)
#     obs.∇τ[1] = (1/(2 * τ[1]^2)) * (tr(obs.V) - dot(obs.storage_n1, obs.storage_n1))
#     # # ldiv!(obs.storage_nn, Vchol, obs.I_n)
#     # # obs.∇τ[1] = (1/(2 * τ[1]^2)) * (tr(obs.storage_nn) - dot(obs.storage_n1, obs.storage_n1))
# end


# # old old code for gradient wrt L
# ldiv!(obs.storage_nq, Vchol, obs.Z)
# # Original code ----
# # copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.Z, obs.storage_nq))
# # # BLAS.gemm!('T', 'N', obs.Z, obs.storage_nq, false, obs.storage_qq)
# # rmul!(obs.storage_qq, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
# # obs.∇L .= - obs.storage_qq
# # New code ----
# BLAS.gemm!('T', 'N', -1., obs.Z, obs.storage_nq, false, obs.∇L)
# rmul!(obs.∇L, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
# # copyto!(obs.storage_1q, BLAS.gemm('T', 'N', reshape(obs.res, (n, 1)), obs.storage_nq))
# BLAS.gemv!('T', 1., obs.storage_nq, obs.res, false, obs.storage_1q)
# # copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.storage_1q, obs.storage_1q))
# # copyto!(obs.storage_qq, BLAS.gemm('T', 'N', reshape(obs.storage_1q, (1, q)), reshape(obs.storage_1q, (1, q))))
# # Since we initialized storage_qq as 0, the following should work
# obs.storage_qq .= 0.
# BLAS.ger!(1., obs.storage_1q, obs.storage_1q, obs.storage_qq)
# rmul!(obs.storage_qq, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
# obs.∇L .+= obs.storage_qq 


"""
init_MoM(m)
Initialize model parameters by the method of moments (MoM). 
For β, the MoM estimator is the same as the OLS estimator.
"""
function init_MoM!(
    m::blblmmModel{T}
    ) where T <: BlasReal

    # OLS for β
    # accumulate sufficient statistics X'y
    xty = zeros(T, m.p) 
    for i in eachindex(m.data)
        BLAS.gemv!('T', one(T), m.data[i].X, m.data[i].y, one(T), xty)
        # gemv!(tA, alpha, A, x, beta, y) 
        # Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA. 
        # alpha and beta are scalars. Return the updated y.
    end
    # print("m.XtX = ", m.XtX, "\n")
    # least square solution for β
    ldiv!(m.β, cholesky(Symmetric(m.XtX)), xty)
    # ldiv!(Y, A, B) -> Y
    # Compute A \ B in-place and store the result in Y, returning the result.

    # For τ 
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
        rss += abs2(norm(m.data[i].res))
    end
    m.τ[1] = m.ntotal / rss # τ is the inverse of error variance
    # we used the inverse so that the objective function is convex

    # MoM Σ
    m.Σ .= 0
    for i in eachindex(m.data)
        # plainly translating the expression
        m.Σ .+= 
            LinearAlgebra.inv(m.data[i].ztz) * 
            transpose(m.data[i].Z) * 
            m.data[i].res *
            transpose(m.data[i].res) *
            m.data[i].Z *
            LinearAlgebra.inv(m.data[i].ztz) .- 
            (1 / m.τ[1]) .* LinearAlgebra.inv(m.data[i].ztz)

        # # inverse of ztz
        # copyto!(m.data[i].storage_qq, LinearAlgebra.inv(m.data[i].ztz))
        # copyto!(m.Σ, - (1 / m.τ[1]) .* m.data[i].storage_qq)
        # # calculate z(ztz)^-1
        # mul!(m.data[i].storage_nq, m.data[i].Z, m.data[i].storage_qq)
        # # calculate rt * z(ztz)^-1
        # # transpose(m.data[i].res) is a vector. this may create trouble.
        # mul!(m.data[i].storage_1q, transpose(m.data[i].res), m.data[i].storage_nq)
        # # copyto!(storage_1q, BLAS.gemm('Y', 'N', m.data[i].res, m.data[i.Z]))
        # # can use gemv
        # mul!(m.data[i].storage_qq, transpose(m.data[i].storage_1q), m.data[i].storage_1q)
        # m.Σ .+= m.data[i].storage_qq
    end
    # print("init_MoM m.Σ = ", m.Σ, "\n")

    # m.β, m.Σ 
end



# """
# init_β(m)
# Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
# squares solution.
# """
# function init_β!(
#     m::blblmmModel{T}
#     ) where T <: BlasReal
#     # accumulate sufficient statistics X'y
#     xty = zeros(T, m.p) 
#     for i in eachindex(m.data)
#         BLAS.gemv!('T', one(T), m.data[i].X, m.data[i].y, one(T), xty)
#         # gemv!(tA, alpha, A, x, beta, y) 
#         # Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA. 
#         # alpha and beta are scalars. Return the updated y.
#     end
#     # print("m.XtX = ", m.XtX, "\n")
#     # least square solution for β
#     ldiv!(m.β, cholesky(Symmetric(m.XtX)), xty)
#     # ldiv!(Y, A, B) -> Y
#     # Compute A \ B in-place and store the result in Y, returning the result.

#     # accumulate residual sum of squares
#     rss = zero(T)
#     for i in eachindex(m.data)
#         update_res!(m.data[i], m.β)
#         rss += abs2(norm(m.data[i].res))
#     end
#     m.τ[1] = m.ntotal / rss # τ is the inverse of error variance
#     # we used the inverse so that the objective function is convex
#     m.β
# end

# """
# Sweep operator
# Only upper triangular part is read and modified.
# """
# function sweep!(
#     A::AbstractMatrix, 
#     k::Integer, 
#     p::Integer=size(A, 2); 
#     inv::Bool=false
# )
#     piv = 1 / A[k, k] # pivot
#     # update entries other than k-th row and column
#     @inbounds for j in 1:p
#         j == k && continue
#         akjpiv = j > k ? A[k, j] * piv : A[j, k] * piv
#         for i in 1:j
#             i == k && continue
#             aik = i > k ? A[k, i] : A[i, k]
#             A[i, j] -= aik * akjpiv
#         end
#     end
#     # update entries of k-th row and column
#     multiplier = inv ? -piv : piv
#     @inbounds for i in 1:k-1
#         A[i, k] *= multiplier
#     end
#     @inbounds for j in k+1:p
#         A[k, j] *= multiplier
#     end
#     # update (k, k)-entry
#     @inbounds A[k, k] = -piv
#     # A
# end

# function sweep!(
#     A::AbstractMatrix, 
#     ks::AbstractVector{<:Integer}, 
#     p::Integer=size(A, 2);
#     inv::Bool=false,
#     need_logdet::Bool=false,
#     check::Bool=false
# )
#     logdetA = 0
#     if need_logdet
#         for k in ks
#             if log(A[k, k]) == -Inf
#                 if check == true
#                     error("Matrix is singular.")
#                 end
#                 logdetA = -Inf
#                 return logdetA
#             end
#             logdetA += log(A[k, k])
#             sweep!(A, k, p, inv=inv)
#         end
#         return logdetA
#     else
#         for k in ks
#             sweep!(A, k, p, inv=inv)
#         end
#     end
# end


const AMat = AbstractMatrix
const AVec = AbstractVector


"""
    sweep!(A, k ; inv=false)
    sweep!(A, ks; inv=false)
Perform the sweep operation (or inverse sweep if `inv=true`) on matrix `A` on element `k`
(or each element in `ks`).  Only the upper triangle is read/swept.
# Example:
    x = randn(100, 10)
    xtx = x'x
    sweep!(xtx, 1)
    sweep!(xtx, 1, true)
"""
function sweep!(A::AMat, k::Integer, inv::Bool = false)
    sweep_with_buffer!(Vector{eltype(A)}(undef, size(A, 2)), A, k, inv)
end

function sweep_with_buffer!(akk::AVec{T}, A::AMat{T}, k::Integer, inv::Bool = false) where
        {T<:BlasFloat}
    # ensure @inbounds is safe
    p = checksquare(A)
    p == length(akk) || throw(DimensionError("incorrect buffer size"))
    @inbounds d = one(T) / A[k, k]  # pivot
    # get column A[:, k] (hack because only upper triangle is available)
    for j in 1:k
        @inbounds akk[j] = A[j, k]
    end
    for j in (k+1):p
        @inbounds akk[j] = A[k, j]
    end
    BLAS.syrk!('U', 'N', -d, akk, one(T), A)  # everything not in col/row k
    rmul!(akk, d * (-one(T)) ^ inv)
    for i in 1:(k-1)  # col k
        @inbounds A[i, k] = akk[i]
    end
    for j in (k+1):p  # row k
        @inbounds A[k, j] = akk[j]
    end
    @inbounds A[k, k] = -d  # pivot element
    A
end

function sweep!(A::AMat{T}, ks::AVec{I}, inv::Bool = false; need_logdet::Bool=false, check::Bool=false) where {T<:BlasFloat, I<:Integer}
    akk = zeros(T, size(A, 1))
    # for k in ks
    #     sweep_with_buffer!(akk, A, k, inv)
    # end
    logdetA = 0
    if need_logdet
        for k in ks
            # print("A[k, k] = ", A[k, k], "\n")
            if A[k, k] < 0
                logdetA = -Inf 
                return logdetA
            elseif log(A[k, k]) == -Inf
                if check == true
                    error("Matrix is singular.")
                end
                logdetA = -Inf 
                return logdetA
            end
            logdetA += log(A[k, k])
            sweep_with_buffer!(akk, A, k, inv)
        end
        return logdetA
    else
        for k in ks
            sweep_with_buffer!(akk, A, k, inv)
        end
    end
    A
end

function sweep_with_buffer!(akk::AVec{T}, A::AMat{T}, ks::AVec{I}, inv::Bool = false) where
        {T<:BlasFloat, I<:Integer}
    for k in ks
        sweep_with_buffer!(akk, A, k, inv)
    end
    A
end


# # # Using the sweep operator
    # logdet_V = sweep!(obs.V, 1:size(obs.V, 1); need_logdet = true, check = false)
    # if logdet_V == -Inf
    #     logl = -Inf # set logl to -Inf and return
    #     return logl
    # end
    # LinearAlgebra.copytri!(obs.V, 'U') # Copy the uppertri to lowertri because we only swept the upper tri
    # lmul!(-1, obs.V)
    # mul!(obs.storage_n1, obs.V, obs.res)
    # logl = (-1//2) * (logdet_V + dot(obs.res, obs.storage_n1))
    # # print("logl = ", logl, "\n")
    # if needgrad
    #     # wrt β
    #     # BLAS.gemm!('T', 'N', 1., obs.X, obs.storage_n1, 1., obs.∇β)
    #     copyto!(obs.∇β, vec(BLAS.gemm('T', 'N', obs.X, obs.storage_n1)))
    #     # wrt τ
    #     # obs.∇τ[1] = (1/(2 * τ[1]^2)) * (sum(diag(inv(obs.V))) - transpose(obs.res) * inv(obs.V) * inv(obs.V) * obs.res)
    #     obs.∇τ[1] = (1/(2 * τ[1]^2)) * (LinearAlgebra.tr(obs.V) - dot(obs.storage_n1, obs.storage_n1))
    #     # print("obs.∇τ[1]=", obs.∇τ[1], "\n")
    #     # wrt L
    #     mul!(obs.storage_nq, obs.V, obs.Z) # 
    #     copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.Z, obs.storage_nq))
    #     # BLAS.gemm!('T', 'N', 1., obs.Z, obs.storage_nq, 1., obs.storage_qq)
    #     rmul!(obs.storage_qq, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
    #     obs.∇L .= - obs.storage_qq
    #     copyto!(obs.storage_1q, BLAS.gemm('T', 'N', reshape(obs.res, (n, 1)), obs.storage_nq))
    #     # ??? why do we need reshape here?
    #     # BLAS.gemm!('T', 'N', 1., reshape(obs.res, (n, 1)), obs.storage_nq, 1., obs.storage_1q)
    #     copyto!(obs.storage_qq, BLAS.gemm('T', 'N', obs.storage_1q, obs.storage_1q))
    #     # BLAS.gemm!('T', 'N', 1., obs.storage_1q, obs.storage_1q, 1., obs.storage_qq)
    #     rmul!(obs.storage_qq, ΣL) # Calculate AB, overwriting A. B must be of special matrix type.
    #     obs.∇L .+= obs.storage_qq 
    # end





########################################################################
########################################################################
# No log transformation
########################################################################
########################################################################
# """
#     modelpar_to_optimpar!(m, par)
# Translate model parameters in `m` to optimization variables in `par`.
# """
# function modelpar_to_optimpar!(
#     par::Vector,
#     m::blblmmModel
#     )
#     # p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
#     #print("modelpar_to_optimpar m.β = ", m.β, "\n")
#     copyto!(par, m.β)
#     par[m.p+1] = m.σ2[1] # take log and then exp() later to make the problem unconstrained
#     # print("modelpar_to_optimpar m.β = ", m.Σ, "\n")
    
#     # Since modelpar_to_optimpar is only called once, it's ok to allocate Σchol
#     Σchol = cholesky(Symmetric(m.Σ), Val(false); check = false)
#     # By using cholesky decomposition and optimizing L, 
#     # we transform the constrained opt problem (Σ is pd) to an unconstrained problem. 
#     m.ΣL .= Σchol.L
#     # print("In modelpar_to_optimparm, m.ΣL = ", m.ΣL, "\n")
#     offset = m.p + 2
#     @inbounds for j in 1:m.q
#         # print("modelpar_to_optimpar m.ΣL[j, j] = ", m.ΣL[j, j], "\n")
#         par[offset] = m.ΣL[j, j] # only the diagonal is constrained to be nonnegative
#         offset += 1
#         @inbounds for i in j+1:m.q
#             par[offset] = m.ΣL[i, j]
#             offset += 1
#         end
#     end
#     par
#     # print("modelpar_to_optimpar par = ", par, "\n")
# end

# """
#     optimpar_to_modelpar!(m, par)
# Translate optimization variables in `par` to the model parameters in `m`.
# """
# function optimpar_to_modelpar!(
#     m::blblmmModel, 
#     par::Vector)
#     # print("Called optimpar_to_modelpar \n")
#     # print("At the beginning of optimpar_to_modelpar, m.Σ = ", m.Σ, "\n")
#     # print("At the beginning of optimpar_to_modelpar, par = ", par, "\n")
#     # p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
#     # print("p = ", p, ", q = ", q, "\n")
#     copyto!(m.β, 1, par, 1, m.p)
#     #print("optimpar_to_modelpar par = ", par, "\n")
#     # copyto!(dest, do, src, so, N)
#     # Copy N elements from collection src starting at offset so, 
#     # to array dest starting at offset do. Return dest.
#     m.σ2[1] = par[m.p+1]
#     fill!(m.ΣL, 0)
#     offset = m.p + 2
#     @inbounds for j in 1:m.q
#         m.ΣL[j, j] = par[offset]
#         offset += 1
#         @inbounds for i in j+1:m.q
#             m.ΣL[i, j] = par[offset]
#             offset += 1
#         end
#     end
#     # print("optimpar_to_modelpar m.ΣL = ", m.ΣL, "\n")
#     mul!(m.Σ, m.ΣL, transpose(m.ΣL))
#     # print("optimpar_to_modelpar, After translating optimpar to modelpar, m.Σ = ", m.Σ, "\n")
#     # updates Σchol so that when we call loglikelihood!(), we are passing the updated cholesky
#     # m.Σchol = cholesky(Symmetric(m.Σ), Val(true); check = false)
#     # Σchol = cholesky(Symmetric(m.Σ), Val(true); check = false)
#     # print("optimpar_to_modelpar m.Σ = ", m.Σ, "\n")
#     m
# end

# function MathProgBase.initialize(
#     m::blblmmModel, 
#     requested_features::Vector{Symbol}
#     )
#     for feat in requested_features
#         if !(feat in [:Grad, :Hess])
#         # if !(feat in [:Grad])
#             error("Unsupported feature $feat")
#         end
#     end
# end

# MathProgBase.features_available(m::blblmmModel) = [:Grad, :Hess] #[:Grad]

# function MathProgBase.eval_f(
#     m::blblmmModel, 
#     par::Vector)
#     # print("in eval_f, par = ", par, "\n")
#     optimpar_to_modelpar!(m, par)
#     # print("Inside eval_f \n")
#     # print("m.β = ", m.β, "\n")
#     # print("m.σ2[1] = ", m.σ2[1], "\n")
#     # print("m.Σ = ", m.Σ, "\n")
#     loglikelihood!(m, false, false, false)
# end


# function MathProgBase.eval_grad_f(
#     m::blblmmModel, 
#     grad::Vector, 
#     par::Vector)
#     # p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
#     optimpar_to_modelpar!(m, par)
#     loglikelihood!(m, true, false, false)
#     # gradient wrt β
#     copyto!(grad, m.∇β)
#     # gradient wrt log(σ2)
#     grad[m.p+1] = m.∇σ2[1]
#     offset = m.p + 2
#     # gradient wrt log(diag(L)) and off-diag(L)
#     @inbounds for j in 1:m.q
#         # On the diagonal, gradient wrt log(ΣL[j,j])
#         grad[offset] = m.∇L[j, j] 
#         offset += 1
#         @inbounds for i in j+1:m.q
#             # Off-diagonal, wrt ΣL[i,j]
#             grad[offset] = m.∇L[i, j]
#             offset += 1
#         end
#     end
#     # print("par = ", par, "\n")
#     # print("grad = ", grad, "\n")
#     nothing
# end

# MathProgBase.eval_g(m::blblmmModel, g, par) = nothing
# MathProgBase.jac_structure(m::blblmmModel) = Int[], Int[]
# MathProgBase.eval_jac_g(m::blblmmModel, J, par) = nothing

# function MathProgBase.hesslag_structure(m::blblmmModel)
#     # Get the linear indices of the upper-triangular of the non-zero blocks
#     npar = ◺(m.p) + 1 + ◺(m.q) + ◺(◺(m.q))
#     #       ββ    σ2σ2  σ2vech(L)   vech(L)vech(L) 
#     arr1 = Vector{Int}(undef, npar)
#     arr2 = Vector{Int}(undef, npar)
#     idx = 1
#     # Hβ
#     for j in 1:m.p
#         for i in 1:j
#             arr1[idx] = i
#             arr2[idx] = j
#             idx += 1
#         end
#     end
#     # Hσ2
#     arr1[idx] = m.p + 1
#     arr2[idx] = m.p + 1
#     idx += 1
#     # HL, take the upper triangle
#     for j in (m.p+2):(m.p + 1 + ◺(m.q))
#         for i in (m.p+2):j
#             arr1[idx] = i
#             arr2[idx] = j
#             idx += 1
#         end
#     end
#     # Hσ2L
#     for j in (m.p+2):(m.p + 1 + ◺(m.q))
#         arr1[idx] = m.p + 1 # same row idx as σ2
#         arr2[idx] = j
#         idx += 1
#     end
#     return (arr1, arr2)
# end

# """
#     diag_idx(n::Integer)
# Get the indices of the diagonal elements of a n x n lower triangular matrix.
# """
# function diag_idx(n::Integer)
#     idx = zeros(Int64, n)
#     idx[1] = 1
#     for i in 2:n
#         idx[i] = idx[i-1] + (n - (i-2))
#     end
#     return idx
# end

# function MathProgBase.eval_hesslag(
#     m::blblmmModel, 
#     H::Vector{T},
#     par::Vector{T}, 
#     σ::T, 
#     μ::Vector{T}) where {T}    
#     # l, q◺ = m.l, ◺(m.q)
#     optimpar_to_modelpar!(m, par)
#     # Do we need to evaluate logl here? Since hessian is always evaluated 
#     # after the gradient, can we just evaluate logl once in the gradient step?
#     loglikelihood!(m, true, true, false)
#     idx = 1
#     @inbounds for j in 1:m.p, i in 1:j
#         H[idx] = m.Hβ[i, j]
#         idx += 1
#     end
#     # hessian wrt log(σ2)
#     H[idx] = m.Hσ2[1] 
#     idx += 1
    
#     # Since we took log of the diagonal elements, log(ΣL[j,j])
#     # we need to do scaling as follows
#     # diagidx = diag_idx(m.q)
#     # for (iter, icontent) in enumerate(diagidx)
#     #     # On the diagonal we have hessian wrt log(ΣL[j,j])
#     #     m.HL[icontent, :] = m.HL[icontent, :] * m.ΣL[iter, iter]
#     #     m.HL[:, icontent] = m.HL[:, icontent] * m.ΣL[iter, iter]
#     #     m.Hσ2L[icontent] = m.Hσ2L[icontent] * m.ΣL[iter, iter]
#     # end
#     @inbounds for j in 1:◺(m.q), i in 1:j
#         H[idx] = m.HL[i, j] 
#         idx += 1
#     end
#     @inbounds for j in 1:◺(m.q)
#         H[idx] = m.Hσ2L[j]
#         idx += 1
#         # # On the diagonal, wrt log(σ2) and log(ΣL[j,j]) 
#         # H[idx] = m.Hσ2L[j, j] * m.σ2[1]
#         # idx += 1
#         # # Off-diagonal, wrt log(σ2) and ΣL[i,j]
#         # for i in (j+1):m.q
#         #     H[idx] = m.Hσ2L[i, j] * m.σ2[1]
#         #     idx += 1
#         # end
#     end
#     lmul!(σ, H)
# end


"""
update_res!(obs, β)
Update the residual vector according to `β`.
"""
function update_res!(
    obs::blblmmObs{T}, 
    β::Vector{T}
    ) where T <: BlasReal
    copyto!(obs.res, obs.y)
    BLAS.gemv!('N', -one(T), obs.X, β, one(T), obs.res) # obs.res - obs.X * β = obs.y - obs.X * β
    obs.res
end
function update_res!(
    m::blblmmModel{T}
    ) where T <: BlasReal
    @inbounds for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
    end
    nothing
end



function loglikelihood!(
    obs::blblmmObs{T},
    β::Vector{T},
    σ²::Vector{T}, # inverse of linear regression variance
    Σ::Matrix{T},
    ΣL::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false,
    updateres::Bool = false
    ) where T <: BlasReal

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    # σ²inv = 1 / σ²[1]
    if needgrad
        fill!(obs.∇β, T(0))
        fill!(obs.∇σ², T(0))
        fill!(obs.∇L, T(0))
    end
    if needgrad
        fill!(obs.Hββ, T(0))
        fill!(obs.Hσ²σ², T(0))
        fill!(obs.HLL, T(0))
        fill!(obs.Hσ²L, T(0))
    end
    ###########
    # objective
    ###########
    updateres && update_res!(obs, β)
    Ω = obs.Z * Σ * obs.Z'
    for i in 1:n
        Ω[i, i] += σ²[1]
    end
    Ωinv = inv(Ω)
    logl = n * log(2π) + logdet(Ω) + obs.res' * Ωinv * obs.res
    logl /= -2
    
    # copy!(obs.storage_qq, obs.ztz) 
    # # L'Z'Z
    # BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq)
    # needgrad && copy!(obs.∇L, obs.storage_qq)
    # # σ² L'Z'Z L
    # BLAS.trmm!('R', 'L', 'N', 'N', σ²[1], ΣL, obs.storage_qq)
    # # storage_qq_1 = σ² L'Z'Z L
    # needgrad && copy!(obs.storage_qq_1, obs.storage_qq) 
    # # I + L'Z'ZL
    # @inbounds for i in 1:q
    #     obs.storage_qq[i, i] += T(1)
    # end
    # LAPACK.potrf!('U', obs.storage_qq) # cholesky on I + L'Z'ZL
    # # Z'r
    # BLAS.gemv!('T', T(1), obs.Z, obs.res, T(0), obs.storage_q)
    # # L'Z'r
    # BLAS.trmv!('L', 'T', 'N', ΣL, obs.storage_q)
    # # chol^{-1} L'Z'r
    # # BLAS.trsv!(ul, tA, dA, obs.storage_qq, b)
    # BLAS.trsv!('U', 'T', 'N', obs.storage_qq, obs.storage_q)
    # logl = n * log(2π * σ²inv)
    # @inbounds for i in 1:q
    #     if obs.storage_qq[i, i] <= 0
    #         logl = -Inf
    #         return logl
    #     else 
    #         # Calculate logdet(I + L'Z'ZL) through cholesky.
    #         logl += 2 * log(obs.storage_qq[i, i])    
    #     end
    # end
    # logl += σ²[1] * dot(obs.res, obs.res) - σ²[1]^2 * dot(obs.storage_q, obs.storage_q)
    # logl /= -2

    ###########
    # gradient
    ###########
    if needgrad
        obs.∇β .= obs.X' * Ωinv * obs.res
        obs.∇σ²[1] = tr(Ωinv) - obs.res' * Ωinv * Ωinv * obs.res
        obs.∇σ²[1] /= -2
        obs.storage_q .= obs.Z' * Ωinv * obs.res
        obs.∇L .= - obs.Z' * Ωinv * obs.Z +  obs.storage_q * obs.storage_q'
    #     # currently, storage_q = chol^{-1} L'Z'r
    #     # update storage_q to (I+σ²LZ'ZL)^{-1} L'Z'r
    #     BLAS.trsv!('U', 'N', 'N', obs.storage_qq, obs.storage_q)
    #     # update storage_q to L(I+σ²LZ'ZL)^{-1} L'Z'r
    #     BLAS.trmv!('L', 'N', 'N', ΣL, obs.storage_q)

    #     # wrt β
    #     # First calculate σ²X'r
    #     BLAS.gemv!('T', σ²[1], obs.X, obs.res, T(0), obs.∇β)
    #     # then, obs.∇β = σ²X'r - σ²^2 X'Z L(I+σ²LZ'ZL)^{-1} L'Z'r
    #     BLAS.gemv!('T', -σ²[1]^2, obs.ztx, obs.storage_q, T(1), obs.∇β)

    #     # wrt σ²
    #     # Since we no longer need obs.res, update obs.res to be Ω^{-1} r
    #     # σ²r - σ²^2 Z L(I+σ²LZ'ZL)^{-1} L'Z'r
    #     BLAS.gemv!('N', -σ²[1]^2, obs.Z, obs.storage_q, σ²[1], obs.res)
    #     # To evaluate tr(Ω^{-1}), Calculate (I+σ²LZ'ZL)^{-1} L'Z'ZL 
    #     # Currently, storage_qq_1 = σ² L'Z'Z L, so we multiply 1/σ²[1] (I+σ²LZ'ZL)^{-1}, 
    #     # which is equivalent to two triangular solves
    #     BLAS.trsm!('L', 'U', 'T', 'N', 1/σ²[1], obs.storage_qq, obs.storage_qq_1)
    #     BLAS.trsm!('L', 'U', 'N', 'N', T(1), obs.storage_qq, obs.storage_qq_1)
    #     # print("σ²inv = ", σ²inv, "\n")
    #     # print("tr(obs.storage_qq_1) = ", tr(obs.storage_qq_1), "\n")
    #     if needhess
    #         # calculate the first two terms in (1/2σ²^4) tr(Ω^{-1}Ω^{-1})
    #         obs.Hσ²σ²[1] = (σ²inv^2 / 2) * n - σ²inv^2 * tr(obs.storage_qq_1)
    #     end
    #     obs.∇σ²[1] = (σ²inv / 2) * n - (1//2) * tr(obs.storage_qq_1) - 
    #                 (σ²inv^2 / 2) * dot(obs.res, obs.res)

    #     # Currently, storage_qq is holding the chol factor
    #     # Before it is destroyed, we need to calculate chol^{-1} L'Z'X for the hessian of β
    #     if needhess
    #         copy!(obs.storage_qp, obs.ztx)
    #         # L'Z'X
    #         BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qp)
    #         # chol^{-1} L'Z'X 
    #         BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.storage_qp)
    #     end

    #     # wrt L
    #     # Currently, obs.∇L = L'Z'Z
    #     # Calculate chol^{-1}L'Z'Z and store it in obs.∇L
    #     BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.∇L)
    #     # Currently, storage_qq_1 = (I+σ²LZ'ZL)^{-1} L'Z'ZL 
    #     ## Before destroying storage_qq_1, we need to calculate its product with 
    #     ## itself for the hessian of \tau. Since storage_qq is no longer needed,
    #     ## we overwrite it with the result.
    #     if needhess
    #         BLAS.gemm!('N', 'N', T(1), obs.storage_qq_1, obs.storage_qq_1, T(0), obs.storage_qq)
    #         # print("tr(obs.storage_qq) = ", tr(obs.storage_qq), "\n")
    #         obs.Hσ²σ²[1] += (1//2) * tr(obs.storage_qq)

    #         # Also calculate the hessian for the cross term Hσ²L
    #         # first set Hσ²L = (I+σ²LZ'ZL)^{-1} L'Z'ZL 
    #         copy!(obs.Hσ²L, obs.storage_qq_1)
    #         # then 2σ²(I+σ²LZ'ZL)^{-1} L'Z'ZL - σ²^2(I+σ²LZ'ZL)^{-1} L'Z'ZL(I+σ²LZ'ZL)^{-1} L'Z'ZL 
    #         BLAS.axpby!(-σ²[1]^2, obs.storage_qq, 2*σ²[1], obs.Hσ²L)
    #         # add -I
    #         @inbounds for i in 1:q
    #             obs.Hσ²L[i, i] -= 1
    #         end
    #         # Left multiply by L
    #         BLAS.trmm!('L', 'L', 'N', 'N', T(1), ΣL, obs.Hσ²L)
    #         # The above calculations were all transposed.
    #         # Next we transpose it, and right multiply ztz
    #         BLAS.gemm!('T', 'N', T(1), obs.Hσ²L, obs.ztz, T(0), obs.storage_qq_1)
    #         copy!(obs.Hσ²L, obs.storage_qq_1)
    #         # We should use the lower triangle of Hσ²L
    #     end
    #     # Next we continue with the calculation for ∇L
    #     # Since we no longer need storage_qq_1, we overwrite it with a rank-k update:
    #     # storage_qq_1 = σ²^2 * Z'ZL(I+σ²LZ'ZL)^{-1}L'Z'Z, which is part of Z'Ω^{-1}Z
    #     BLAS.syrk!('U', 'T', σ²[1]^2, obs.∇L, T(0), obs.storage_qq_1) 
    #     # Since syrk only updated the upper triangle, do copytri
    #     LinearAlgebra.copytri!(obs.storage_qq_1, 'U') 
    #     # Update obs.storage_qq_1 as -Z'Ω^{-1}Z = -σ²Z'Z + σ²^2 * Z'ZL(I+σ²LZ'ZL)^{-1}L'Z'Z
    #     BLAS.axpy!(-σ²[1], obs.ztz, obs.storage_qq_1)
    #     # copy it to obs.∇L
    #     copy!(obs.∇L, obs.storage_qq_1)
    #     # Currently res = Ω^{-1} r
    #     # Update storage_q as Z'Ω^{-1} r
    #     BLAS.gemv!('T', T(1), obs.Z, obs.res, T(0), obs.storage_q)
    #     # Rank-1 update of ∇L as ∇L + storage_q storage_q'
    #     BLAS.syr!('U', T(1), obs.storage_q, obs.∇L)
    #     LinearAlgebra.copytri!(obs.∇L, 'U')
    #     # In fact, this is only the gradient wrt Σ
    #     # We will right multiply L once in the aggregation step to save computation
    end

    ###########
    # hessian
    ###########
    if needhess
        obs.Hββ .= obs.X' * Ωinv * obs.X
        obs.Hσ²σ²[1] = tr(Ωinv * Ωinv)
        obs.Hσ²σ²[1] /= 2  
        A = obs.Z' * Ωinv * obs.Z 
        B = A * ΣL
        C = ΣL' * B
        D = [1 0 0; 0 1 0; 0 0 0; 0 0 1] 
        # D is the copy matrix with the third row replaced by 0
        # Kqq = commutation(q, q)
        # HLLtemp = kron(C, A) + kron(B', B) * Kqq 
        # obs.HLL .= D' * HLLtemp * D
        obs.HLL .= Ct_A_kron_B_C(C, A) + Ct_At_kron_A_KC(B)
        obs.Hσ²L .= vec(vec(obs.Z' * Ωinv * Ωinv * obs.Z * ΣL)' * D)
    #     # wrt β
    #     # Currently, storage_qp = chol^{-1} L'Z'X.
    #     copy!(obs.Hββ, obs.xtx)
    #     # Currently, storage_qp = chol^{-1} L'Z'X 
    #     # Update Hββ as -σ²X'X + σ²^2 * X'ZL(I+σ²LZ'ZL)^{-1}L'Z'X through rank-k update
    #     BLAS.syrk!('U', 'T', σ²[1]^2, obs.storage_qp, -σ²[1], obs.Hββ) 
    #     # only the upper triangular of Hββ is updated
        
    #     # wrt σ²
    #     # already calculated above

    #     # wrt vech L 
    #     fill!(obs.HLL, T(0))
    #     # Currently storage_qq_1 = -Z'Ω^{-1}Z, and is Symmetric
    #     # We update it to Z'Ω^{-1}Z, and copy it to storage_qq
    #     lmul!(T(-1), obs.storage_qq_1)
    #     copy!(obs.storage_qq, obs.storage_qq_1)
    #     # print("At Z'Ω^{-1}Z, should be symm, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     # Update storage_qq_1 to be Z'Ω^{-1}Z L
    #     BLAS.trmm!('R', 'L', 'N', 'N', T(1), ΣL, obs.storage_qq_1) 
    #     # print("At Z'Ω^{-1}ZL, shouldnt be symm, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     # Update HLL as C'(L'Z'Ω^{-1}Z ⊗ Z'Ω^{-1}ZL)KC
    #     # print("Z'Ω^{-1}ZL, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     Ct_At_kron_A_KC!(obs.HLL, obs.storage_qq_1)
    #     # print("After Ct_At_kron_A_KC, obs.HLL = ", obs.HLL, "\n")
    #     # Scale by coefficient -1
    #     lmul!(T(-1), obs.HLL)
    #     # Update storage_qq_1 to be L'Z'Ω^{-1}ZL
    #     BLAS.trmm!('L', 'L', 'T', 'N', T(1), ΣL, obs.storage_qq_1)
    #     # print("At L'Z'Ω^{-1}ZL, should be symm, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #     # Scale by coefficient 3
    #     lmul!(T(3), obs.storage_qq_1)
    #     # Update HLL as C'(L'Z'Ω^{-1}Z ⊗ Z'Ω^{-1}ZL)KC + C'(3*L'Z'Ω^{-1}ZL ⊗ Z'Ω^{-1}Z)C
    #     # Currently, storage_qq = Z'Ω^{-1}Z
    #     Ct_A_kron_B_C!(obs.HLL, obs.storage_qq_1, obs.storage_qq)
    #     # print("After Ct_A_kron_B_C, obs.HLL = ", obs.HLL, "\n")
    #     # Temporarily multiply by -1 to make it positive definite. 
    #     # According to the derivation, there should be no such multiplication.
    #     # lmul!(T(-1), obs.HLL)
    #     if any(isnan.(obs.HLL))
    #         print("Z'Ω^{-1}Z, obs.storage_qq = ", obs.storage_qq, "\n")
    #         print("3L'Z'Ω^{-1}ZL, obs.storage_qq_1 = ", obs.storage_qq_1, "\n")
    #         print("obs.HLL = ", obs.HLL, "\n")
    #     end

    #     # wrt Hσ²L
    #     # already calculated above
    end

    logl
end








