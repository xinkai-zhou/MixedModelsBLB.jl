using DataFrames, DataFramesMeta, StatsBase, Query

# The dataframe approach will load the entire dataset into memory.
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),b = repeat([2, 1], outer=[4]),c = 1:8)
g = groupby(df, :a)
# 1. using DataFrames.filter, but it doesn't take groupedDataFrame
# DataFrames.filter(:a => x -> x == 1, groupby(df, :a))

# 2. using DataFramesMeta, but it doesn't allow filtering on the grouping variable
# @where(g, :a == 1)
@where(g, mean(:b) == 1)

# 3. using Query, cannot filter grouped data
df |> @groupby(_.a) |> @filter(_.a == 1)
g = df |> @groupby(_.a) |> @map(obsvec = myobs(_))
# so i have to filter and then group. 