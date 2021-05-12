
# Detailed Usage

This page covers options you will need for dealing with extremely large longitudinal data sets and for forming estimates and confidence intervals of functions of the parameter estimates. 

## Data Input

As shown in the [Quick Tutorial](quick-tutorial.md) section, data can be loaded in memory (RAM) as a ```DataFrame```. However, real-world longitudinal data such as Electronic Meidcal Records (EMR) may be too large to fit in RAM. 

Fortunately, since the Bag of Little Bootstrap (BLB) method operates on subsets rather than the full data set, we do not need to load the full data in RAM. By using the out-of-core functionality of JuliaDB, we stream in subsets that are relevant to the analysis and leave the rest of the data on the disk. 

To illustrate, we simulated a longitudinal data set with 1000 subjects and 10 measurements per subject. Covariates $x_1, x_2$ are simulated from standard normal and $x_3$ is a binary (categorical) variable. To demonstrate how to deal with data sets that are scattered in multiple files, we saved ours in 5 pieces named `File1.csv, ..., File5.csv`. Suppose that the full data is too large to fit in RAM, then we can load the data as follows:


```julia
using MixedModelsBLB, StatsModels, Random, Ipopt, JuliaDB, Tables
```


```julia
dat = JuliaDB.loadtable(glob("../../../test/data/files/*.csv"), output = "bin", distributed = false)
```




    Table with 10000 rows, 5 columns:
    y          x1          x2          x3   id
    ────────────────────────────────────────────
    5.58629    1.19027     0.897602    "M"  1
    6.71752    2.04818     0.206015    "M"  1
    3.90137    1.14265     -0.553413   "M"  1
    3.08093    0.459416    0.422711    "M"  1
    1.10712    -0.396679   2.41808     "M"  1
    -0.724565  -0.664713   -2.00348    "M"  1
    3.87239    0.980968    0.166398    "M"  1
    1.78055    -0.0754831  0.687318    "M"  1
    4.50491    0.273815    0.835254    "M"  1
    1.71213    -0.194229   0.650404    "M"  1
    1.28998    -0.339366   -0.205452   "M"  2
    1.4559     -0.843878   -0.0682929  "M"  2
    ⋮
    6.18278    2.19452     -0.517101   "F"  999
    2.55695    -0.243373   0.792531    "F"  1000
    3.31882    0.752808    0.0390571   "F"  1000
    -1.04349   -0.437674   -0.583502   "F"  1000
    0.111996   -0.532844   -0.716789   "F"  1000
    2.99027    0.512117    1.07059     "F"  1000
    0.266275   -1.26409    0.274989    "F"  1000
    3.32454    -0.0722508  0.426401    "F"  1000
    2.87777    -0.694544   1.85115     "F"  1000
    2.15322    -0.751663   0.667909    "F"  1000
    -0.794344  -2.36031    1.30847     "F"  1000



The option ```output = "bin"``` specifies that we want to load the data to the directory ```"bin"```. If we do not set it, then the data is loaded in RAM by default. Next we run BLB and print the results.


```julia
solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, 
    mehrotra_algorithm = "yes", warm_start_init_point = "yes", 
    warm_start_bound_push = 1e-9)

blb_ests = blb_full_data(
        MersenneTwister(1),
        dat;
        feformula   = @formula(y ~ 1 + x1 + x2 + x3),
        reformula   = @formula(y ~ 1 + x1),
        id_name     = "id", 
        cat_names   = ["x3"], 
        subset_size = 100,
        n_subsets   = 10, 
        n_boots     = 500,
        solver      = solver,
        verbose     = false,
        nonparametric_boot = true
    );
```

    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    



```julia
print(blb_ests)
```

    Bag of Little Boostrap (BLB) for linear mixed models.
    Number of subsets: 10
    Number of grouping factors per subset: 200
    Number of bootstrap samples per subset: 200
    Confidence interval level: 95%
    
    Variance Components parameters
    ─────────────────────────────────────────────────
                       Estimate    CI Lower  CI Upper
    ─────────────────────────────────────────────────
    (Intercept)       0.96871     0.882979   1.05727
    x1                0.988777    0.892928   1.07907
    (Intercept) : x1  0.0419663  -0.0239626  0.108318
    Residual          1.02973     0.999208   1.06251
    ─────────────────────────────────────────────────
    
    Fixed-effect parameters
    ──────────────────────────────────────────
                 Estimate   CI Lower  CI Upper
    ──────────────────────────────────────────
    (Intercept)  1.0479     0.954289  1.14302
    x1           1.02489    0.960579  1.08892
    x2           1.01025    0.988965  1.03265
    x3: M        0.030941  -0.102612  0.163412
    ──────────────────────────────────────────

## Parallel Processing

Compared to the traditional bootstrap method, which needs to repeatedly access the full data during the resampling step, BLB is much easier to parallelize because once subsets are taken, the resampling step only needs to access the small subsets. As a result, if our machine has $N$ cores, then we can process $N$ subsets in parallel to drastically speed up the analysis. 

This can be achieved by first adding worker nodes as follows:


```julia
using Distributed
addprocs(2) # Use addprocs(N) if your machine has N cores
@everywhere using MixedModelsBLB
workers()
```




    4-element Array{Int64,1}:
     2
     3
     7
     8



Then we run BLB with the following code. Note that it is the same code as above because `blb_full_data()` will automatically make use of the worker nodes that we have just made available.


```julia
solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, 
    mehrotra_algorithm = "yes", warm_start_init_point = "yes", 
    warm_start_bound_push = 1e-9)

blb_ests = blb_full_data(
        MersenneTwister(1),
        dat;
        feformula   = @formula(y ~ 1 + x1 + x2 + x3),
        reformula   = @formula(y ~ 1 + x1),
        id_name     = "id", 
        cat_names   = ["x3"], 
        subset_size = 100,
        n_subsets   = 10, 
        n_boots     = 500,
        solver      = solver,
        verbose     = false,
        nonparametric_boot = true
    );
```

    wks_schedule = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3]
          From worker 2:	
          From worker 2:	******************************************************************************
          From worker 2:	This program contains Ipopt, a library for large-scale nonlinear optimization.
          From worker 2:	 Ipopt is released as open source code under the Eclipse Public License (EPL).
          From worker 2:	         For more information visit https://github.com/coin-or/Ipopt
          From worker 2:	******************************************************************************
          From worker 2:	
          From worker 3:	
          From worker 3:	******************************************************************************
          From worker 3:	This program contains Ipopt, a library for large-scale nonlinear optimization.
          From worker 3:	 Ipopt is released as open source code under the Eclipse Public License (EPL).
          From worker 3:	         For more information visit https://github.com/coin-or/Ipopt
          From worker 3:	******************************************************************************
          From worker 3:	



```julia
print(blb_ests)
```

    Bag of Little Boostrap (BLB) for linear mixed models.
    Number of subsets: 10
    Number of grouping factors per subset: 200
    Number of bootstrap samples per subset: 200
    Confidence interval level: 95%
    
    Variance Components parameters
    ───────────────────────────────────────────────
                      Estimate   CI Lower  CI Upper
    ───────────────────────────────────────────────
    (Intercept)       0.97719   0.891017   1.06351
    x1                1.1049    0.996983   1.21472
    (Intercept) : x1  0.106918  0.0379043  0.175623
    Residual          1.0097    0.979607   1.03991
    ───────────────────────────────────────────────
    
    Fixed-effect parameters
    ────────────────────────────────────────────
                  Estimate    CI Lower  CI Upper
    ────────────────────────────────────────────
    (Intercept)  1.05446     0.962949   1.14765
    x1           0.941411    0.875429   1.00897
    x2           0.999205    0.976363   1.0208
    x3: M        0.0289467  -0.0967177  0.155865
    ────────────────────────────────────────────

Note that the results are slightly different from above. This is because using multiple workers affect the random seeds used for subsetting and resampling, so the difference is due to sampling variability and will become smaller if we increase the number of subsets or the number of bootstrap samples.

## Customized Confidence Intervals

If you are interested in getting the confidence intervals of some functions of the parameters, you can construct it using the estimates stored in the output of `blb_full_data()`. 

To illustrate, suppose we want to calculate the 95% confidence interval of the Intra-class Correlation Coefficient (ICC). This can be done by calculating the 95% percentile CIs from all subsets and then average them across subsets. 


```julia
using LinearAlgebra, StatsBase
icc   = zeros(200, 10)
level = 0.95

# Calculate ICC
for j in 1:10
    for i in 1:200
        # ICC = σa² / (σa² + σe²)
        icc[i, j] = blb_ests.all_estimates[j].Σs[:, :, i][1, 1] / sum(diag(blb_ests.all_estimates[j].Σs[:, :, i]))
    end
end
```


```julia
# Calculate the 95% CIs on 10 subsets
CIs = zeros(10, 2)
for j in 1:10
    CIs[j, :] = StatsBase.percentile(icc[:, j], 100 * [(1 - level) / 2, 1 - (1-level) / 2])
end
```


```julia
# Calculate the BLB CI by averaging CIs across subsets
mean(CIs, dims = 1)
```




    1×2 Array{Float64,2}:
     0.435465  0.500777



## Tips

### Ipopt

By setting a higher `print_level`, you may notice that Ipopt performs lots of line searches. One way to remedy it and to speed up your analysis is to set `mehrotra_algorithm="yes"`, which disables line search. The option `mu_strategy="adaptive"` may also be helpful.

### Categorical Variables

To make sure that we do not miss any values of a categorical variable in a subset, `blb_full_data()` performs checking once a subset is taken. If a subset fails to contain certain values, then a new subset is taken and this step is repeated until we find a valid subset. 

This works as long as all values are relatively common. If a certain value is scarce, however, then it may take a long time to get a valid subset. In such cases, we recommend grouping the values into fewer categories. 


```julia

```
