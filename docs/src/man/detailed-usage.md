
# Detailed Usage

This page covers options you will need for dealing with extremely large longitudinal data sets and for forming estimates and confidence intervals of functions of the parameter estimates. 

## Data Input

As shown in the [Quick Tutorial](quick-tutorial.md) section, data can be loaded in memory (RAM) as a ```DataFrame```. However, real-world longitudinal data such as Electronic Meidcal Records (EMR) may be too large to fit in RAM. 

Fortunately, since the Bag of Little Bootstrap (BLB) method operates on subsets rather than the full data set, we do not need to load the full data in RAM. By interfacing with a database, we stream in subsets that are relevant to the analysis and leave the rest of the data on the hard disk. 

To illustrate, we created a MySQL database called `MixedModelsBLB` on the local host and imported a simulated longitudinal data set with 1000 subjects and 20 measurements per subject to the `testdata` table. Covariates $x_1, x_2, x_3, z_1$ are simulated from standard normal. 

By providing a connection object, the `blb_db` function can interface with the database and only fetch data subsets that are relevant to the analysis. 


```julia
using MixedModelsBLB, StatsModels, Random, Ipopt, DBInterface, MySQL, DataFrames
```


```julia
con = DBInterface.connect(MySQL.Connection, "127.0.0.1", "USERNAME", "PASSWORD"; db = "MixedModelsBLB");
```


```julia
# Show the first 10 rows of the data set
DBInterface.execute(con,  "SELECT * FROM testdata LIMIT 10;") |> DataFrame
```




<table class="data-frame"><thead><tr><th></th><th>id</th><th>y</th><th>x1</th><th>x2</th><th>x3</th><th>z1</th></tr><tr><th></th><th>Int32</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>10 rows × 6 columns</p><tr><th>1</th><td>1</td><td>-1.74387</td><td>-1.72976</td><td>-1.28905</td><td>-1.47062</td><td>-0.267067</td></tr><tr><th>2</th><td>1</td><td>1.23021</td><td>0.795949</td><td>-0.33527</td><td>-0.535211</td><td>1.49908</td></tr><tr><th>3</th><td>1</td><td>0.495366</td><td>0.670062</td><td>0.0704676</td><td>-0.963544</td><td>0.797304</td></tr><tr><th>4</th><td>1</td><td>1.79272</td><td>0.550852</td><td>0.341794</td><td>-1.38511</td><td>-0.17164</td></tr><tr><th>5</th><td>1</td><td>3.33667</td><td>-0.0633746</td><td>1.73517</td><td>0.1343</td><td>-0.46908</td></tr><tr><th>6</th><td>1</td><td>4.35921</td><td>1.33694</td><td>1.29992</td><td>-0.616117</td><td>0.217624</td></tr><tr><th>7</th><td>1</td><td>3.05776</td><td>-0.0731486</td><td>0.206364</td><td>-1.71999</td><td>0.359146</td></tr><tr><th>8</th><td>1</td><td>-0.493603</td><td>-0.745464</td><td>-1.00886</td><td>0.320769</td><td>0.320025</td></tr><tr><th>9</th><td>1</td><td>-1.31595</td><td>-1.22006</td><td>-0.850056</td><td>-1.44737</td><td>0.259216</td></tr><tr><th>10</th><td>1</td><td>-0.446968</td><td>-0.0531773</td><td>1.12941</td><td>-0.492271</td><td>0.459696</td></tr></tbody></table>




```julia
solver = Ipopt.IpoptSolver(print_level=0, max_iter=100, 
    mehrotra_algorithm = "yes", warm_start_init_point = "yes", 
    warm_start_bound_push = 1e-9)

blb_ests = blb_db(
        MersenneTwister(1),
        con,
        "testdata",
        feformula   = @formula(y ~ 1 + x1 + x2 + x3),
        reformula   = @formula(y ~ 1 + z1),
        id_name     = "id", 
        cat_names   = Vector{String}(), 
        subset_size = 200,
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
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    



```julia
print(blb_ests)
```

    Bag of Little Boostrap (BLB) for linear mixed models.
    Number of subsets: 10
    Number of grouping factors per subset: 200
    Number of bootstrap samples per subset: 500
    Confidence interval level: 95%
    
    Variance Components parameters
    ─────────────────────────────────────────────────
                        Estimate   CI Lower  CI Upper
    ─────────────────────────────────────────────────
    (Intercept)       0.964051     0.881758  1.04793
    z1                3.1184       2.87197   3.37536
    (Intercept) : z1  0.00680443  -0.104531  0.120664
    Residual          1.46487      1.43443   1.4963
    ─────────────────────────────────────────────────
    
    Fixed-effect parameters
    ─────────────────────────────────────────
                 Estimate  CI Lower  CI Upper
    ─────────────────────────────────────────
    (Intercept)  0.981893  0.919381   1.04176
    x1           1.01022   0.991362   1.02862
    x2           1.0158    0.997535   1.03386
    x3           0.989067  0.971957   1.00621
    ─────────────────────────────────────────

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
