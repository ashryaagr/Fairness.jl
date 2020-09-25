"""
Helper function for ReweighingWrapper and ReweighingSamplingWrapper.
grps is an array of values of protected attribute. y is an array of ground truth values.
Array of (Frequency) weights are returned from this function.
"""
function _calculateWeights(grps_fair, y_fair, grps_train, y_train)
    grps_classes = levels(grps_train)
    n_classes = length(grps_classes)
    n_train = length(y_train)
    n_fair = length(y_fair)

    fav_fair = y_fair.==levels(y_fair)[2] # Boolean
    unfav_fair = y_fair.==levels(y_fair)[1]
    n₊_fair = sum(fav_fair) # Total Number of favourable outcomes

    fav_train = y_train.==levels(y_train)[2] # Boolean
    unfav_train = y_train.==levels(y_train)[1]
    n₊_train = sum(fav_train) # Total Number of favourable outcomes

    weights = zeros(Float64, n_train)
    for i in 1:n_classes
        class_train = grps_train.==grps_classes[i]
        class_fair = grps_fair.==grps_classes[i]
        weights[class_train .& fav_train] .= n₊_fair*sum(class_fair)/(n_fair*sum(class_fair .& fav_fair))
        weights[class_train .& unfav_train] .= (n_fair-n₊_fair)*sum(class_fair)/(n_fair*sum(class_fair .& unfav_fair)) end

    weights[isnan.(weights)].=1
    return weights
end

"""
    ReweighingWrapper

ReweighingWrapper is a preprocessing algorithm wrapper in which Weights for each group-label combination is calculated.
These calculated weights are then passed to the classifier model which further uses it to make training fair.
"""
struct ReweighingWrapper{M<:MLJBase.Model} <: DeterministicComposite
    grp::Symbol
    classifier::M
    holdout::ResamplingStrategy
end

"""
    ReweighingWrapper(classifier=nothing, grp=:class)

Instantiates a ReweighingWrapper which wrapper the `classifier` with the Reweighing fairness algorithm.
The sensitive attribute can be specified by the parameter `grp`.
If `classifier` doesn't support weights while training, an error is thrown.
"""
function ReweighingWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class, holdout::ResamplingStrategy=Holdout(fraction_train=0.8))
    model = ReweighingWrapper(grp, classifier, holdout)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::ReweighingWrapper)
    warning = ""
    model.classifier!=nothing || (warning *= "No classifier specified in model\n")
    target_scitype(model) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
    supports_weights(model.classifier) || (warning *= "Classifier provided does not support weights\n")
    return warning
end

function MLJBase.fit(model::ReweighingWrapper,
    verbosity::Int, X, y)
    indices = MLJBase.train_test_pairs(model.holdout, 1:length(y))

    X_train, y_train = X[indices[1][1], :], y[indices[1][1]]
    X_fair, y_fair = X[indices[1][2], :], y[indices[1][2]]
    weights = _calculateWeights(X_fair[:, model.grp], y_fair, X_train[:, model.grp], y_train)

    Xs = source(X_train)
    ys = source(y_train)
    Ws = source(weights)

    classifier = model.classifier
    mach1 = machine(classifier, Xs, ys, Ws)
    ŷ = MMI.predict(mach1, Xs)

    mach2 = machine(Deterministic(), Xs, ys; predict=ŷ)
    return!(mach2, model, verbosity)
end

MMI.input_scitype(::Type{<:ReweighingWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:ReweighingWrapper{M}}) where M = AbstractVector{<:Finite{2}}

"""
    ReweighingSamplingWrapper

ReweighingSamplingWrapper is a preprocessing algorithm wrapper in which Weights for each group-label combination is calculated.
Using the calculated weights, rows are sampled uniformly. The weight is used to sample uniformly.
The number of datapoints used to train after sampling from the reweighed dataset can be controlled by `factor`.
"""
struct ReweighingSamplingWrapper{M<:MLJBase.Model} <: DeterministicComposite
    grp::Symbol
    classifier::M
    factor::Float64
    rng::Union{Int,AbstractRNG}
    holdout::ResamplingStrategy
end

"""
    ReweighingSamplingWrapper(classifier=nothing, grp=:class, factor=1, rng=Random.GLOBAL_RNG)

Instantiates a ReweighingSamplingWrapper which wrapper the classifier with the Reweighing fairness algorithm together with sampling.
The sensitive attribute can be specified by the parameter `grp`.
`factor`*number_of_samples_in_original_data datapoints are sampled using calculated weights and then used to train after sampling from the reweighed dataset.
A negative or no value value for `factor` parameter instructs the algorithm to use the same number of datapoints as in original sample.
"""
function ReweighingSamplingWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class, factor::Float64=1.0, rng=nothing, holdout::ResamplingStrategy=Holdout(fraction_train=0.8))
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if rng == nothing
        rng = Random.GLOBAL_RNG
    end
    model = ReweighingSamplingWrapper(grp, classifier, factor, rng, holdout)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::ReweighingSamplingWrapper)
    warning = ""
    model.classifier!=nothing || (warning *= "No classifier specified in model\n")
    target_scitype(model) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
    return warning
end

function MLJBase.fit(model::ReweighingSamplingWrapper,
    verbosity::Int, X, y)

    indices = MLJBase.train_test_pairs(model.holdout, 1:length(y))

    X_train, y_train = X[indices[1][1], :], y[indices[1][1]]
    X_fair, y_fair = X[indices[1][2], :], y[indices[1][2]]
    weights = _calculateWeights(X_fair[:, model.grp], y_fair, X_train[:, model.grp], y_train)

    n = model.factor<0 ? length(y) : Int(round(model.factor*length(y)))
    indices = sample(model.rng, 1:length(y_train), FrequencyWeights(weights), n; replace=true, ordered=false)
    Xnew = X_train[indices, :]
    ynew = y_train[indices]
    Xs = source(Xnew)
    ys = source(ynew)
    mach1 = machine(model.classifier, Xs, ys)
    ŷ = MMI.predict(mach1, Xs)

    mach2 = machine(Deterministic(), Xs, ys; predict=ŷ)
    return!(mach2, model, verbosity)
end

MMI.input_scitype(::Type{<:ReweighingSamplingWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:ReweighingSamplingWrapper{M}}) where M = AbstractVector{<:Finite{2}}
