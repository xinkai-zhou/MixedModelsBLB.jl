
# MixedModelsBLB.jl Documentation

MixedModelsBLB.jl is a Julia package for analyzing massive longitudinal data using Linear Mixed Models (LMMs) through the Bag of Little Bootstrap (BLB) method. 


## Package Feature

- Lightning fast compared to the traditional bootstrap-based LMM analysis.
- Compatible with a variety of data inputs:
    - Supports inputs that [integrate with the ```Tables.jl``` interface](https://github.com/JuliaData/Tables.jl/blob/main/INTEGRATIONS.md), inclusing the commonly used ```DataFrames.jl```.
    - Supports data sets that are too large to fit into the memory. 
- Supports parallel processing.
- Supports both gradient-based and gradient-free solvers such as [Ipopt](https://github.com/coin-or/Ipopt), [Mosek](https://github.com/MOSEK/Mosek.jl), and [Nlopt](https://github.com/JuliaOpt/NLopt.jl) through the [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) interface. 



## Manual Outline

```@contents
Pages = [
    "man/installation.md",
    "man/quick-tutorial.md",
    "man/contributing.md",
    "man/api.md",
]
Depth = 2
```