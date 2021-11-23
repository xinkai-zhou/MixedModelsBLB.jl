
# Installation

This package requires Julia v1.6 or later, which can be obtained from https://julialang.org/downloads/.

To install the package, start Julia and type

```
using Pkg
Pkg.add(url = "https://github.com/xinkai-zhou/MixedModelsBLB.jl")
```

This does not install any solvers. If you don't have a solver installed already, you will want to install a solver such as ```Ipopt``` by running

```
Pkg.add("Ipopt")
```
