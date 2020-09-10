### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ c365f2f2-f361-11ea-39ae-799739acc8eb
begin
	using Pkg
	Pkg.activate("synbench_env", shared=true)
	#Pkg.add("Fairness")
	#Pkg.add("MLJBase")
	#Pkg.add("MLJ")
	#Pkg.add("StatsPlots")
	#Pkg.add("DecisionTree")
	#Pkg.add("DataFrames")
	using Fairness, MLJ, MLJBase
	using DataFrames, StatsPlots
end

# ╔═╡ f3b75fa4-f361-11ea-36f5-fdef03dddf63
# Score on training data to assess model quality
function get_scores(m, X, y, rows, set)
  yhat = predict(m, rows=rows)
  if typeof(yhat[1]) <: MLJBase.UnivariateFinite
    yhat = MLJBase.mode.(yhat)
  end
  ft = fair_tensor(yhat, y[rows], X[rows, :z])
  df = disparity(
    [accuracy, false_positive_rate],
    ft, refGrp="A",
    func=(x, y)->(x-y)
  )
  df[:,"accuracy"] .= accuracy(ft)
  df[:,"false_positive_rate"] .= false_positive_rate(ft)
  df[:,"set"] .= set
  return df
end

# ╔═╡ f4db5732-f361-11ea-3359-b72ac75cebb4
function run_synthetic_experiment(md, ntrain = 10000, genFun = genZafarData; genargs...)
  (method, mod) = md

  ntest = 1000
  repls = 30

  # Train data
  n = ntrain + (ntest * repls)
  X,y = genFun(n; genargs...)

  # Fit classifier
  train, test = partition(eachindex(y), ntrain/n, shuffle=true)
  m = machine(mod, X, y)
  fit!(m, rows=train)

  # Compute scores on train data
  df = get_scores(m, X, y, train, "insample")

  # Score on different test subsets
  tsts = [test[i:(i+ntest-1)] for i in 1:ntest:length(test)]
  for tst in tsts
    append!(df, get_scores(m, X, y, tst, "oob"))
  end
  df = df[df.labels .== "B",:]
  select!(df, Not(1))
  df[:, "n"] .= ntrain
  df[:, "method"] .= method
  return df
end

# ╔═╡ f5a67cc8-f361-11ea-0de5-33599ea9d491
# Create a list of classifiers to compare
begin
	@load RandomForestClassifier pkg=DecisionTree
	model = @pipeline ContinuousEncoder RandomForestClassifier
	m1 = ReweighingSamplingWrapper(classifier=model, grp=:z)
	m2 = LinProgWrapper(classifier=m1, grp=:z, measure=false_positive_rate)
	m3 = EqOddsWrapper(classifier=m1, grp=:z)
	clf = Dict("RF" => model, "RF_RW" => m1, "RF_RW_LP" => m2, "RF_RW_EO" => m3)
	keys(clf)
end

# ╔═╡ f65f0eb4-f361-11ea-19d8-d976d9779175
begin
	# Test on Zafar et al., 2017 Dataset
	res = DataFrame()
	for c in clf
    	append!(res, run_synthetic_experiment(c, 20000))
	end
end

# ╔═╡ 1ee229b6-f362-11ea-2ab7-671cb85e12f6
begin
	@df res[res.set .!= "insample",:] scatter(:accuracy,  :false_positive_rate_disparity, group = :method)
	xlabel!("Accuracy")
	ylabel!("Δ False Positive Rate")
end

# ╔═╡ 1fbbd274-f362-11ea-079a-35c51d9bd87c
begin
	res2 = DataFrame()
	for c in clf
    	append!(res2, run_synthetic_experiment(c, 10000, genBiasedSampleData; sampling_bias = 0.5))
	end
end

# ╔═╡ 5e2a5706-f362-11ea-0781-1d3f16ee8376
begin
	@df res2[res2.set .!= "insample",:] scatter(:accuracy,  :false_positive_rate_disparity, group = :method)
	xlabel!("Accuracy")
	ylabel!("Δ False Positive Rate")
end

# ╔═╡ Cell order:
# ╠═c365f2f2-f361-11ea-39ae-799739acc8eb
# ╠═f3b75fa4-f361-11ea-36f5-fdef03dddf63
# ╠═f4db5732-f361-11ea-3359-b72ac75cebb4
# ╠═f5a67cc8-f361-11ea-0de5-33599ea9d491
# ╠═f65f0eb4-f361-11ea-19d8-d976d9779175
# ╠═1ee229b6-f362-11ea-2ab7-671cb85e12f6
# ╠═1fbbd274-f362-11ea-079a-35c51d9bd87c
# ╠═5e2a5706-f362-11ea-0781-1d3f16ee8376
