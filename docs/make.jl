using Documenter, MixedModelsBLB

ENV["DOCUMENTER_DEBUG"] = "true"

makedocs(
    format = Documenter.HTML(),
    sitename = "MixedModelsBLB",
    modules = [MixedModelsBLB]
)

deploydocs(
    repo   = "https://github.com/xinkai-zhou/MixedModelsBLB.jl.git",
    target = "build"
)