
# Quick Tutorial

Consider the ```sleepstudy``` dataset (Belenky et al., 2003), which is from a sleep deprivation study that measured the average reaction time from 18 subjects over 10 days.


```julia
using MixedModelsBLB, CSV, DataFrames, StatsModels, Random, Ipopt
```


```julia
sleepstudy = CSV.read("../../../test/data/sleepstudy.csv", DataFrame);
first(sleepstudy, 5)
```




<div class="data-frame"><p>5 rows × 3 columns</p><table class="data-frame"><thead><tr><th></th><th>Reaction</th><th>Days</th><th>id</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Int64">Int64</th></tr></thead><tbody><tr><th>1</th><td>249.56</td><td>0</td><td>308</td></tr><tr><th>2</th><td>258.705</td><td>1</td><td>308</td></tr><tr><th>3</th><td>250.801</td><td>2</td><td>308</td></tr><tr><th>4</th><td>321.44</td><td>3</td><td>308</td></tr><tr><th>5</th><td>356.852</td><td>4</td><td>308</td></tr></tbody></table></div>



To fit a Linear Mixed Model (LMM) of the form

```math
$$\text{Reaction} ~ \text{Days} + (1|\text{ID})},$$
```

and perform statistical inference using the Bag of Little Bootstraps (BLB), we use the following code:


```julia
blb_ests = blb_full_data(
        MersenneTwister(1),
        sleepstudy;
        feformula   = @formula(Reaction ~ 1 + Days),
        reformula   = @formula(Reaction ~ 1),
        id_name     = "id", 
        cat_names   = Array{String,1}(), 
        subset_size = 10,
        n_subsets   = 20, 
        n_boots     = 500,
        method      = :ML,
        solver      = Ipopt.IpoptSolver(print_level=0),
        verbose     = false,
        nonparametric_boot = true
    );
```

    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    


In this chunk,

- ```MersenneTwister(1)``` sets the random seed for subsetting and resampling.
- ```feformula``` and ```reformula``` specify the fixed and random effect formula, respectively.
- ```id_name``` is the name of the grouping factor such as subject ID.
- ```cat_names``` is a vector of categorical variable names. If there aren't any, simply set it as we did above. 
- ```subset_size```, ```n_subsets```, and ```n_boots``` are BLB parameters. Typically, we recommend setting
    - ```subset_size``` = $N^{0.6}$ or $N^{0.7}$, where $N$ is the total number of subjects. 
    - ```n_subsets``` = 10-20.
    - ```n_boots```   = 500-2000
- [Ipopt](https://github.com/coin-or/Ipopt) is a freely-available gradient-based solver and works quite well. [Knitro](https://www.artelys.com/solvers/knitro/) is 3-5 times faster than Ipopt but requires a liscense (you might be eligible for an academic liscense).



To see the result, type


```julia
print(blb_ests)
```

    Bag of Little Boostrap (BLB) for linear mixed models.
    Method: ML
    Number of subsets: 20
    Number of grouping factors per subset: 10
    Number of bootstrap samples per subset: 500
    Confidence interval level: 95%
    
    Variance Components parameters
    ─────────────────────────────────────────
                 Estimate  CI Lower  CI Upper
    ─────────────────────────────────────────
    (Intercept)  1289.92    463.5     2187.22
    Residual      855.397   541.333   1227.79
    ─────────────────────────────────────────
    
    Fixed-effect parameters
    ──────────────────────────────────────────
                 Estimate   CI Lower  CI Upper
    ──────────────────────────────────────────
    (Intercept)  248.978   235.696    262.032
    Days          10.4608    7.73577   13.2131
    ──────────────────────────────────────────

Results are displayed in two tables, showing the BLB estimates and confidence intervals for both fixed effect and variance components parameters. 

## Reference

Gregory Belenky, Nancy J. Wesensten, David R. Thorne, Maria L. Thomas, Helen C. Sing, Daniel P. Redmond, Michael B. Russo and Thomas J. Balkin (2003) Patterns of performance degradation and restoration during sleep restriction and subsequent recovery: a sleep dose-response study. Journal of Sleep Research 12, 1–12.
