function get_df(fp::FairnessProblem, models_fn)

	Logging.disable_logging(LogLevel(0))

	#aliasing
	X = fp.task.X
	y = fp.task.y
	measures=fp.measures
	protected_attr = fp.task.grp
	debiasmeasures = fp.task.debiasmeasures
	refGrp = fp.refGrp
	runs=fp.repls
	nfolds=fp.nfolds
	random_seed = fp.seed
	models, model_names = models_fn(fp)

	Logging.disable_logging(LogLevel(0))
	# Here runs implies number of runs of n_folds cv-folds
	df = DataFrame([String[], Float64[],
			[Float64[] for i in 1:length(measures)]..., String[], Int[], Int[]],
			["model", "accuracy",
			[string(measure)*"_disparity" for measure in measures]...,
			"domain", "fold", "replication"]) #domain tells whether result is for in-sample or out-sample
	Random.seed!(random_seed)
	seeds = abs.(rand(Int, runs))
	if model_names==nothing model_names = string.(models) end
	progress = Progress(runs*length(models)*nfolds, 1)
	for i in 1:runs
		cv = StratifiedCV(nfolds=nfolds, shuffle=true, rng=seeds[i])
		tr_tt_pairs = MLJBase.train_test_pairs(cv, 1:length(y),
			categorical(string.(X[!, protected_attr]) .* "-" .* string(y)))
		for i_model in 1:length(models)
			model = models[i_model]
			for j in 1:nfolds
				Random.seed!(seeds[i])
				mach = machine(model, X, y)
				@suppress fit!(mach, rows=tr_tt_pairs[j][1], verbosity=0)

				# Add out-sample performance measures
				ŷ = MMI.predict(mach, rows=tr_tt_pairs[j][2])
				if typeof(ŷ[1])<:MLJBase.UnivariateFinite ŷ = StatsBase.mode.(ŷ) end
				ft = fair_tensor(ŷ, y[tr_tt_pairs[j][2]],
										X[tr_tt_pairs[j][2], protected_attr])
				accVal = accuracy(ft)
				# disps = disparity(measures, ft, refGrp=refGrp)
				# For now, the value of Disparity I will consider will be:
				# (Overall Fairness Value)/(Fairness value for reference Group)
				push!(df, [model_names[i_model], accVal,
				[measure(ft, grp="0")/measure(ft, grp="1") for measure in measures]...,
				"test", j, i])


				# Add in-sample performance measures
				ŷ = MMI.predict(mach, rows=tr_tt_pairs[j][1])
				if typeof(ŷ[1])<:MLJBase.UnivariateFinite ŷ = StatsBase.mode.(ŷ) end
				ft = fair_tensor(ŷ, y[tr_tt_pairs[j][1]],
										X[tr_tt_pairs[j][1], protected_attr])
				accVal = accuracy(ft)
				# disps = disparity(measures, ft, refGrp=refGrp)
				# For now, the value of Disparity I will consider will be:
				# (Overall Fairness Value)/(Fairness value for reference Group)
				push!(df, [model_names[i_model], accVal,
				[measure(ft, grp="0")/measure(ft, grp="1") for measure in measures]...,
				"train", j, i])
				next!(progress)
			end
		end
	end
	CSV.write(joinpath(pwd(), fp.name*"-results.csv"), df)
	return df
end

function get_pareto_df(fp::FairnessProblem, models_fn, alphas=0:0.1:1)

	Logging.disable_logging(LogLevel(0))

	#aliasing
	X = fp.task.X
	y = fp.task.y
	measures=fp.measures
	protected_attr = fp.task.grp
	debiasmeasures = fp.task.debiasmeasures
	refGrp = fp.refGrp
	runs=fp.repls
	nfolds=fp.nfolds
	random_seed = fp.seed
	models, model_names = models_fn(fp, 0.5)


	# Here runs implies number of runs of n_folds cv-folds
	df = DataFrame([String[], Float64[], Float64[],
			[Float64[] for i in 1:length(measures)]..., String[], Int[], Int[]],
			["model", "alpha", "accuracy",
			[string(measure)*"_disparity" for measure in measures]...,
			"domain",  "fold", "replication"])
			#domain tells whether result is for in-sample or out-sample
	Random.seed!(random_seed)
	seeds = abs.(rand(Int, runs))
	if model_names==nothing model_names = string.(models) end
	progress = Progress(length(alphas)*runs*length(models)*nfolds, 1)
	for alpha in alphas
		models, _ = models_fn(fp, alpha)
		for i in 1:runs
			cv = StratifiedCV(nfolds=nfolds, shuffle=true, rng=seeds[i])
			tr_tt_pairs = MLJBase.train_test_pairs(cv, 1:length(y),
				categorical(string.(X[!, protected_attr]) .* "-" .* string(y)))
			for i_model in 1:length(models)
				model = models[i_model]
				for j in 1:nfolds
					Random.seed!(seeds[i])
					mach = machine(model, X, y)
					@suppress fit!(mach, rows=tr_tt_pairs[j][1])
					# Add out-sample performance measures
					ŷ = MMI.predict(mach, rows=tr_tt_pairs[j][2])
					if typeof(ŷ[1])<:MLJBase.UnivariateFinite ŷ = StatsBase.mode.(ŷ) end
					ft = fair_tensor(ŷ, y[tr_tt_pairs[j][2]],
											X[tr_tt_pairs[j][2], protected_attr])
					accVal = accuracy(ft)
					# disps = disparity(measures, ft, refGrp=refGrp)
					# For now, the value of Disparity I will consider will be:
					# (Overall Fairness Value)/(Fairness value for reference Group)
					push!(df, [model_names[i_model], alpha, accVal,
					[measure(ft, grp="0")/measure(ft, grp="1") for measure in measures]...,
					"test", j, i])


					# Add in-sample performance measures
					ŷ = MMI.predict(mach, rows=tr_tt_pairs[j][1])
					if typeof(ŷ[1])<:MLJBase.UnivariateFinite ŷ = StatsBase.mode.(ŷ) end
					ft = fair_tensor(ŷ, y[tr_tt_pairs[j][1]],
											X[tr_tt_pairs[j][1], protected_attr])
					accVal = accuracy(ft)
					# disps = disparity(measures, ft, refGrp=refGrp)
					# For now, the value of Disparity I will consider will be:
					# (Overall Fairness Value)/(Fairness value for reference Group)
					push!(df, [model_names[i_model], alpha, accVal,
					[measure(ft, grp="0")/measure(ft, grp="1") for measure in measures]...,
					"train", j, i])
					next!(progress)
				end
			end
		end
	end
	CSV.write(joinpath(pwd(), fp.name*"-pareto-results.csv"), df)
	return df
end

# Core idea behind this function is to get the output after evaluation, hypothesis testing, etc. that can be
# directly passed to plotting functions or any benchmark related function without need to pass any other argument.
"""
    fairevaluate(classifiers, X, y; measures=nothing, measure=nothing, grp=:class, priv_grps, random_seed=12345, n_grps=6)

Performed paired t-test for each pair of classifier in classifiers and return p values and t statistics.
# Arguments
- `classifiers`: Array of classifiers to compare
- `X`: DataFrame with features and protected attribute
- `y`: Binary Target Variable
- `measures=nothing`: The measures to be evaluated and used for HypothesisTests.
                If this is not specified, the `measure` argument is used
- `measure=nothing`: The performance/fairness measure used to perform hypothesis tests.
                    If no values for measure is passed, then Disparate Impact will be used by default.
- `grp=:class`: Protected Attribute Name
- `priv_grps=nothing`: If default measure i.e. Disparate Impact is used, then pass an array of groups which are privileged in dataset.
- `random_seed=12345`: Random seed to ensure reproducibility
- `n_grps=6`: Number of folds for cross validation
# Returns
A dictionary with following keys vs values is returned
- `measures`: names of the measures
- `classifier_names`: names of the classifiers. If a pipeline is used, it will show pipeline and associated number.
- `results`: 3-dimensional array with evaluation result. Its size is measures x classifiers x fold_number.
- `pvalues`: 3-dimensional array with pvalues for each pair of classifier. Its size is measures x classifiers x classifiers.
- `tstats`:3-dimensional array with tstats for each pair of classifier. Its size is measures x classifiers x classifiers.
"""
function fairevaluate(
    classifiers::Array{<:MLJBase.Model,1}, X, y;
    measure = nothing,
    measures = nothing,
    grp = :class,
    priv_grps = nothing,
    random_seed::Int = 12345,
    n_folds = 6,
    classifier_names = nothing
)
    Random.seed!(random_seed)
    y = coerce(y, OrderedFactor)

    @assert(!(measure==nothing && measures==nothing && priv_grps==nothing))
    if priv_grps!=nothing
        @assert(all(
            priv_grp in levels(X[!, grp]) for priv_grp in priv_grps
            for classifier in classifiers
        ))
        measure = measure == nothing ? DisparateImpact(grp, priv_grps) : measure
    end
    @assert(all(
        target_scitype(classifier) <: AbstractVector{<:Finite}
        for classifier in classifiers
    ))

    n = length(classifiers)

    if measures==nothing measures=[measure] end
    n_measures = length(measures)

    for i in 1:n_measures
        if typeof(measures[i]) <: MetricWrapper
            measures[i].grp = grp
        end
    end

    results = zeros(n_measures, n, n_folds)
    cv = CV(nfolds = n_folds, shuffle=false, rng=random_seed)
    for i = 1:n
        Random.seed!(random_seed)
        operation = istype(classifiers[i], Probabilistic) ? MLJBase.predict_mode : MLJBase.predict
        result = evaluate(
            classifiers[i],
            X,
            y,
            resampling = cv,
            measures = measures,
            operation = operation,
            verbosity=0,
        )
        for j in 1:n_measures
            results[j, i, :] = result.per_fold[j]
        end
    end
    pvalues, tstats = zeros(n_measures, n, n), zeros(n_measures, n, n)
    for k in 1:n_measures
        for i = 1:n
            for j = 1:n
                ttestResult = OneSampleTTest(results[k, i, :], results[k, j, :])
                tstats[k, i, j] = ttestResult.t
                pvalues[k, i, j] = pvalue(ttestResult)
            end
        end
    end
    if classifier_names == nothing
        classifier_names = string.(classifiers)
    end
    dict = Dict()
    dict["measures"] = string.(measures)
    dict["classifier_names"] = classifier_names
    dict["results"] = results
    dict["pvalues"] = pvalues
    dict["tstats"] = tstats
    return dict
end
