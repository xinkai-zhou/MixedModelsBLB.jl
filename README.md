![LOGO](https://github.com/xinkai-zhou/MixedModelsBLB.jl/blob/master/logo-text.png)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xinkai-zhou.github.io/MixedModelsBLB.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xinkai-zhou.github.io/MixedModelsBLB.jl/stable)
[![CI](https://github.com/xinkai-zhou/MixedModelsBLB.jl/workflows/CI/badge.svg)](https://github.com/xinkai-zhou/MixedModelsBLB.jl/actions)
[![Codecov](https://codecov.io/gh/xinkai-zhou/MixedModelsBLB.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/xinkai-zhou/MixedModelsBLB.jl)



MixedModelsBLB.jl is a Julia package for analyzing massive longitudinal data using Linear Mixed Models (LMMs) through the Bag of Little Bootstrap (BLB) method. It offers significant speedup compared to the traditional bootstrap-based LMM analysis. See [documentation](https://xinkai-zhou.github.io/MixedModelsBLB.jl/dev) for more details. 

## Installation
Download and install Julia. Within Julia, copy and paste the following:
```
using Pkg
pkg "add https://github.com/xinkai-zhou/MixedModelsBLB.jl.git"
```
This package supports Julia v1.6+ for Mac, Linux, and window machines.

## Citation
If you use MixedModelsBLB.jl, please cite the following paper: 

X. Zhou, J. J. Zhou, and H. Zhou, Bag of little bootstraps for massive and distributed longitudinal data, Stat. Anal. Data Min.: ASA Data Sci. J. (2021), 1–8. https://doi.org/10.1002/sam.11563