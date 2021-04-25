
# Quick Tutorial

Consider the ```sleepstudy``` dataset (Belenky et al., 2003), which is from a sleep deprivation study that measured the average reaction time from 18 subjects over 10 days.


```julia
using MixedModelsBLB, CSV, StatsModels, Random, Ipopt
```


```julia
sleepstudy = CSV.read("../../../test/data/sleepstudy.csv");
sleepstudy[1:10, :]
```




<table class="data-frame"><thead><tr><th></th><th>Reaction</th><th>Days</th><th>id</th></tr><tr><th></th><th>Float64</th><th>Int64</th><th>Int64</th></tr></thead><tbody><p>10 rows × 3 columns</p><tr><th>1</th><td>249.56</td><td>0</td><td>308</td></tr><tr><th>2</th><td>258.705</td><td>1</td><td>308</td></tr><tr><th>3</th><td>250.801</td><td>2</td><td>308</td></tr><tr><th>4</th><td>321.44</td><td>3</td><td>308</td></tr><tr><th>5</th><td>356.852</td><td>4</td><td>308</td></tr><tr><th>6</th><td>414.69</td><td>5</td><td>308</td></tr><tr><th>7</th><td>382.204</td><td>6</td><td>308</td></tr><tr><th>8</th><td>290.149</td><td>7</td><td>308</td></tr><tr><th>9</th><td>430.585</td><td>8</td><td>308</td></tr><tr><th>10</th><td>466.353</td><td>9</td><td>308</td></tr></tbody></table>



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
        solver      = Ipopt.IpoptSolver(print_level=0),
        verbose     = false,
        nonparametric_boot = true
    );
```

In this chunk,

- ```MersenneTwister(1)``` sets the random seed for subsetting and resampling.
- ```feformula``` and ```reformula``` specify the fixed and random effect formula, respectively.
- ```id_name``` is the name of the grouping factor such as subject ID.
- ```cat_names``` is a vector of categorical variable names. If there aren't any, simply set it as we did above. 
- ```subset_size```, ```n_subsets```, and ```n_boots``` are BLB parameters. Typically, we recommend setting
    - ```subset_size``` = $N^{0.6}$ or $N^{0.7}$, where $N$ is the total number of subjects. 
    - ```n_subsets``` = 10-20.
    - ```n_boots```   = 500-2000
- [Ipopt](https://github.com/coin-or/Ipopt) is a freely-available gradient-based solver and works quite well. [Mosek](https://www.mosek.com) is 3-5 times faster than ```Ipopt``` but requires a liscense (you might be eligible for an [academic liscense](https://www.mosek.com/products/academic-licenses/)).

To see the result, type


```julia
print(blb_ests)
```

    Bag of Little Boostrap (BLB) for linear mixed models.
    Number of subsets: 20
    Number of grouping factors per subset: 10
    Number of bootstrap samples per subset: 500
    Confidence interval level: 95%
    
    Variance Components parameters
    ─────────────────────────────────────────
                 Estimate  CI Lower  CI Upper
    ─────────────────────────────────────────
    (Intercept)  1016.66    338.074   1752.28
    Residual      938.435   575.239   1367.96
    ─────────────────────────────────────────
    
    Fixed-effect parameters
    ──────────────────────────────────────────
                 Estimate   CI Lower  CI Upper
    ──────────────────────────────────────────
    (Intercept)  253.803   241.482     265.762
    Days          10.4451    7.72668    13.102
    ──────────────────────────────────────────

Results are displayed in two tables, showing the BLB estimates and confidence intervals for both fixed effect and variance components parameters. 

## Reference

Gregory Belenky, Nancy J. Wesensten, David R. Thorne, Maria L. Thomas, Helen C. Sing, Daniel P. Redmond, Michael B. Russo and Thomas J. Balkin (2003) Patterns of performance degradation and restoration during sleep restriction and subsequent recovery: a sleep dose-response study. Journal of Sleep Research 12, 1–12.
