include("utils.jl")

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
