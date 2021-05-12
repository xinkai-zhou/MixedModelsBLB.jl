using Documenter, MixedModelsBLB

# ENV["DOCUMENTER_DEBUG"] = "true"

# makedocs(
#     format = Documenter.HTML(),
#     sitename = "MixedModelsBLB",
#     modules = [MixedModelsBLB]
# )

# deploydocs(
#     repo   = "https://github.com/xinkai-zhou/MixedModelsBLB.jl.git",
#     target = "build"
# )

# using Documenter: Documenter, makedocs, deploydocs
# using PkgTemplates: PkgTemplates

makedocs(;
    modules=[MixedModelsBLB],
    authors="Xinkai Zhou, Hua Zhou",
    repo="https://github.com/xinkai-zhou/MixedModelsBLB.jl.git/blob/{commit}{path}#{line}",
    sitename="MixedModelsBLB.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xinkai-zhou.github.io/MixedModelsBLB.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "man/installation.md",
        "Quick Tutorial" => "man/quick-tutorial.md",
        "Detailed Usage" => "man/detailed-usage.md",
        "Contributing" => "man/contributing.md",
        "API" => "man/api.md",
    ],
)

deploydocs(;
    repo="github.com/xinkai-zhou/MixedModelsBLB.jl",
)