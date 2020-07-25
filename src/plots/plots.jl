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
