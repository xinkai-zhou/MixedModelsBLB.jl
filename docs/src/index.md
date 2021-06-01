
# MixedModelsBLB.jl Documentation

MixedModelsBLB.jl is a Julia package for analyzing massive longitudinal data using Linear Mixed Models (LMMs) through the Bag of Little Bootstrap (BLB) method. 


## Package Feature

- Lightning fast compared to the traditional bootstrap-based LMM analysis.
- Compatible with a variety of data inputs:
    - Supports inputs that [integrate with the ```Tables.jl``` interface](https://github.com/JuliaData/Tables.jl/blob/main/INTEGRATIONS.md), inclusing the commonly used ```DataFrames.jl```.
    - Supports interfacing with databases, which is ideal for data sets that exceed the machine's memory limit.
- Supports parallel processing.
- Supports both gradient-based and gradient-free solvers such as [Ipopt](https://github.com/coin-or/Ipopt), [Knitro](https://github.com/jump-dev/KNITRO.jl), and [Nlopt](https://github.com/JuliaOpt/NLopt.jl) through the [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) interface. 



## Manual Outline

```@contents
Pages = [
    "man/installation.md",
    "man/quick-tutorial.md",
    "man/detailed-usage.md",
    "man/contributing.md",
    "man/api.md",
]
Depth = 2
```