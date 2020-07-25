function accuracy_vs_fairness(result)
	@assert result.measure[1].measure==accuracy
	@assert length(result.measure)==2
	n_folds = length(result.per_fold[1])
	grps = collect(keys(result.per_fold[2][1]))
	n_grps = length(grps)
	accVal = zeros(n_grps, n_folds)
	fairVals = zeros(n_grps, n_folds)
	plot(title="Accuracy vs Fairness", xlabel="accuracy", ylabel="fairness")
	for i in 1:n_grps
		for j in 1:n_folds
			accVal[i, j] = result.per_fold[1][j][grps[i]]
			fairVals[i, j] = result.per_fold[2][j][grps[i]]
		end
		plot!(accVal[i, :], fairVals[i, :], label=grps[i])
	end
	display(plot!())
end

function algorithm_comparison(algorithms, X, y; measure, grp::Symbol=:class)
	grps = X[!, grp]
	categories = levels(grps)
	train, test = partition(eachindex(y), 0.7, shuffle=true)
	plot(title="Algorithm Fairness Comparison", seriestype=:scatter, xlabel="accuracy", ylabel="fairness")
	for i in length(algorithms)
		mach = machine(algorithms[i], X, y)
		fit!(mach, rows=train)
		ŷ = predict(mach, rows=test)
		if typeof(ŷ) <: UnivariateFiniteArray
			ŷ = StatsBase.mode.(ŷ)
		end
		ft = fair_tensor(ŷ, y[test], X[test, grp])
		plot!([accuracy(ft)], [measure(ft)], seriestype=:scatter, label="algorithm_$i")
	end
	display(plot!())
end
