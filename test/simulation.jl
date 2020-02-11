
module simulation

println()
@info "Running simulation.jl"

using Random, Distributions, LinearAlgebra, DelimitedFiles
using MixedModelsBLB

println()
@info "simulate dataset"
# Simulate dataset
Random.seed!(1)
N = 500 # number of individuals
p = 1
q = 1
reps = 5 # number of observations from each individual
y = 1 .+ # fixed intercept
    repeat(rand(Normal(0, 1), N), inner = reps) + # random intercept, standard normal
    rand(Normal(0, 1), reps * N); # error, standard normal
X = fill(1., (reps * N, p));
Z = fill(1., (reps * N, q));
id = repeat(1:N, inner = reps);

println()
@info "set BLB parameters and pre-allocate arrays"
# BLB parameters
s = 2  # number of BLB subsets
r = 10 # number of monte carlo iterations
b = Int64(floor(N^0.9)) # subset size
blb_id = fill(0, b) # preallocate an array to store BLB sample indices
ns = zeros(b) # preallocate an array to store multinomial counts
# pre-allocate subset estimates
β_b = zeros(p)
Σ_b = Matrix{Float64}(undef, q, q)
τ_b = Vector{Float64}(undef, 1)
logls = zeros(s * r)
# pre-allocate arrays to store result
β̂ = [Vector{Float64}(undef, p) for i = 1:(s * r)]
# If we use fill(Vector{Float64}(undef, p), s*r), then all elements of this vector
# refer to the same empty vector. So if we use copyto!() to update, all elements of 
# the vector will be updated with the same value.
Σ̂ = [Matrix{Float64}(undef, q, q) for i = 1:(s * r)]
τ̂ = zeros(0)

println()
@info "start BLB procedure"
for j = 1:s
    # Subsetting
    # take a subsample of size b w/o replacement
    # ?? Here we should sample from the unique IDs.
    sample!(unique(id), blb_id; replace = false)
    sort!(blb_id) # sort the ids
    # construct blblmmObs and blblmmModel for the subset
    obs = Vector{blblmmObs{Float64}}(undef, b)
    for (i, grp) in enumerate(blb_id)
        gidx = id .== grp
        yi = Float64.(y[gidx])
        Xi = Float64.(X[gidx, :])
        Zi = Float64.(Z[gidx, :])
        # copyto!(yb, y[id .∈ blb_id])
        # copyto!(Xb, X[id .∈ blb_id, :])
        # copyto!(Zb, Z[id .∈ blb_id, :])
        # ?? does this avoid allocating to arrays such as yi, Xi?
        # obs[i] = blblmmObs(Float64.(y[gidx]), loat64.(X[gidx, :]), Float64.(Z[gidx, :]))
        obs[i] = blblmmObs(yi, Xi, Zi)
    end
    m = blblmmModel(obs) # Construct the blblmmModel type
    
    # initialize model parameters
    init_β!(m) # initalize β and τ using least squares   
    # print("initial estimate", m.β, "\n")
    m.Σ .= Diagonal(0.5 .* ones(size(obs[1].Z, 2))) # initialize Σ with identity
    
    #  m.β
    #  m.Σ
    #  m.τ

    # Fit LMM using the subsample and get parameter estimates
    fit!(m) 
    # print("subset estimate", m.β, "\n")
    copyto!(β_b, m.β)
    copyto!(Σ_b, m.Σ)
    copyto!(τ_b, m.τ)
    
    println()
    @info  "Begin Bootstrapping on subset" j
    # Bootstrapping
    for k = 1:r
        # Generate a parametric bootstrap sample of y and update m
        # Update y in place by looping over blblmmModel
        println()
        @info  "Bootstrap iteration" k
        for bidx = 1:b
            copyto!(
                m.data[bidx].y, 
                m.data[bidx].X * β_b + # fixed effect
                m.data[bidx].Z * rand(MvNormal(zeros(size(Σ_b, 1)), Σ_b)) + # random effect
                rand(Normal(0, sqrt(1 / τ_b[1])), length(m.data[bidx].y)) # error, standard normal
            )
        end

        # print(m.data[1].y)
        # update_w!(m, ones(b))
        # fit!(m)
        # print("BLB est", m.β, "\n")
        
        # get weights by drawing N i.i.d. samples from multinomial
        rand!(Multinomial(N, ones(b)/b), ns) 
        
        # update weights in blblmmModel
        update_w!(m, ns)
        
        # use weighted loglikelihood to fit the bootstrapped dataset
        fit!(m)
        
        # extract estimates
        i = (j-1) * r + k # the index for storage purpose
        copyto!(β̂[i], m.β)
        copyto!(Σ̂[i], m.Σ) # copyto!(Σ̂[i], m.Σ) doesn't work. how to index into a vector of matrices?
        push!(τ̂, m.τ[1]) # ?? better ways to do this?
        
    end
    # Aggregate over the r Monte Carlo iterations
end
# Aggregate over the s BLB subsets


# fit model using NLP on profiled loglikelihood
@show β̂
@show Σ̂
@show τ̂

writedlm("beta-hat.csv", β̂, ',')
writedlm("sigma-hat.csv", Σ̂, ',')
writedlm("tau-hat.csv", τ̂, ',')

end # module

