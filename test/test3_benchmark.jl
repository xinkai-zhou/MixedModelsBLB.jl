module blbBenchmark

using InteractiveUtils, LinearAlgebra, Profile, Random
using BenchmarkTools, MixedModelsBLB
using DelimitedFiles, Random, CSV

Random.seed!(123)

# 


gcm = GaussianCopulaVCModel(gcs)

@info "Initial point:"
init_β!(gcm)
@show gcm.β
fill!(gcm.Σ, 1)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ
# @btime update_Σ!(gcm) setup=(fill!(gcm.Σ, 1))

@show loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.Σ, true, false)
# @code_warntype loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.Σ, true, false)
@btime loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.Σ, true, false)

@show loglikelihood!(gcm, true, false)
@show [gcm.∇β; gcm.∇τ; gcm.∇Σ]
# @code_warntype loglikelihood!(gcm, false, false)
# @code_llvm loglikelihood!(gcm, false, false)
@btime loglikelihood!(gcm, true, false)

# Solvers:
# Ipopt.IpoptSolver(print_level=0)
# NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_CCSAQ, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND_RESTART, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_RESTART, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_VAR1, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_VAR2, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LN_COBYLA, maxeval=10000)
# NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)

@info "MLE:"
solver = Ipopt.IpoptSolver(print_level=0)
@time fit!(gcm, solver)
@show [gcm.β; gcm.τ; gcm.Σ]
@show [gcm.∇β; gcm.∇τ; gcm.∇Σ]
@show loglikelihood!(gcm)
# @btime fit!(gcm, solver) setup=(init_β!(gcm); 
#     standardize_res!(gcm); update_quadform!(gcm, true); fill!(gcm.Σ, 1);
#     update_Σ!(gcm))

# Profile.clear()
# @profile begin
for solver in [
    NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    NLopt.NLoptSolver(algorithm=:LN_COBYLA, maxeval=10000),
    NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000),
    NLopt.NLoptSolver(algorithm=:LD_CCSAQ, maxeval=4000),
    NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000),
    NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000),
    # NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND_RESTART, maxeval=4000),
    #NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND, maxeval=4000),
    #NLopt.NLoptSolver(algorithm=:LD_TNEWTON_RESTART, maxeval=4000),
    #NLopt.NLoptSolver(algorithm=:LD_TNEWTON, maxeval=4000),
    # NLopt.NLoptSolver(algorithm=:LD_VAR1, maxeval=4000),
    # NLopt.NLoptSolver(algorithm=:LD_VAR2, maxeval=4000),
    # Ipopt.IpoptSolver(print_level=0)
    ]
    println()
    @show solver
    # re-set starting point
    init_β!(gcm)
    fill!(gcm.Σ, 1)
    update_Σ!(gcm)
    # fit 
    fit!(gcm, solver)
    @time fit!(gcm, solver)
    @show loglikelihood!(gcm)
    @show [gcm.β; gcm.τ; gcm.Σ]
    @show [gcm.∇β; gcm.∇τ; gcm.∇Σ]
    println()
end
# end
# Profile.print(format=:flat)

end