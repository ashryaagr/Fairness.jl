### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 49331584-f33c-11ea-08df-352d2b92dd45
begin
	using Pkg
	Pkg.activate("../examples")
	using Fairness, MLJ, PrettyPrinting
	using DataFrames, StatsPlots
end

# ╔═╡ 9988c856-f33c-11ea-1b99-73cf8d4b49aa
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

# ╔═╡ 9a65ce88-f33c-11ea-3eb0-0f125503a23c
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

# ╔═╡ af3ab6b6-f33c-11ea-2f05-83cfaad088dd
# Create a list of classifiers to compare
begin
	@load RandomForestClassifier pkg=DecisionTree
	model = @pipeline ContinuousEncoder RandomForestClassifier
	m1 = ReweighingSamplingWrapper(classifier=model, grp=:z)
	m2 = LinProgWrapper(classifier=m1, grp=:z, measure=false_positive_rate)
	m3 = EqOddsWrapper(classifier=m1, grp=:z)
	clf = Dict("RF" => model, "RF_RW" => m1, "RF_RW_RP" => m2, "RF_RW_EO" => m3)
end

# ╔═╡ db7e28ae-f33c-11ea-3d57-df9a9aff35da
begin
	res = DataFrame()
	for c in clf
    	append!(res, run_synthetic_experiment(c, 1000))
	end
end

# ╔═╡ eabbfe3a-f345-11ea-38d5-1b6a34f6877d
begin
	@df res scatter(:accuracy, :false_positive_rate_disparity, group = [:method,:set])
	xlabel!("Accuracy")
	ylabel!("Δ False Positive Rate")
end	

# ╔═╡ dbc43cd4-f33c-11ea-145b-2d901f2554b2
begin
	@df res[res.set .!= "insample",:] scatter(:accuracy, :false_positive_rate_disparity, group = :method)
	xlabel!("Accuracy")
	ylabel!("Δ False Positive Rate")
end	

# ╔═╡ a4112d50-f342-11ea-12fb-c5d5e2f55815
begin
	b = 0
	a = (1,2)
	(b,c) = a
	b
end

# ╔═╡ 0cdac786-f345-11ea-01fd-7d0c2f31b8c6
  begin
	ntrain = 1000
	ntest = 1000
	repls = 30
	mod=m3
	# Train data
	n = ntrain + (ntest * repls)
	X,y = genZafarData(n)

	# Fit classifier
	train, test = partition(eachindex(y), ntrain/n, shuffle=true)
	m = machine(mod, X, y)
	fit!(m, rows=train)
	rows=train
	yhat = predict(m, rows=rows)
	#ft = fair_tensor(yhat, y[rows], X[rows, :z])
	#df = disparity(
	#[accuracy, false_positive_rate],
	#ft, refGrp="A",
	#func=(x, y)->(x-y)
	#)
end

# ╔═╡ 24fbdc4e-f345-11ea-1ac8-13a4fac43b20
mod

# ╔═╡ Cell order:
# ╠═49331584-f33c-11ea-08df-352d2b92dd45
# ╠═9988c856-f33c-11ea-1b99-73cf8d4b49aa
# ╠═9a65ce88-f33c-11ea-3eb0-0f125503a23c
# ╠═af3ab6b6-f33c-11ea-2f05-83cfaad088dd
# ╠═db7e28ae-f33c-11ea-3d57-df9a9aff35da
# ╠═eabbfe3a-f345-11ea-38d5-1b6a34f6877d
# ╠═dbc43cd4-f33c-11ea-145b-2d901f2554b2
# ╠═a4112d50-f342-11ea-12fb-c5d5e2f55815
# ╠═0cdac786-f345-11ea-01fd-7d0c2f31b8c6
# ╠═24fbdc4e-f345-11ea-1ac8-13a4fac43b20
