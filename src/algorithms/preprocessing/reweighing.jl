"""
Helper function for ReweighingWrapper and ReweighingSamplingWrapper.
grps is an array of values of protected attribute. y is an array of ground truth values.
Array of (Frequency) weights are returned from this function.
"""
function _calculateWeights(grps, y)
    grps_classes = levels(grps)
    n_classes = length(grps_classes)
    n = length(y)

    fav = y.==levels(y)[2] # Boolean
    unfav = y.==levels(y)[1]
    n₊ = sum(fav) # Total Number of favourable outcomes

    weights = zeros(Float64, n)
    for i in 1:n_classes
        class = grps.==grps_classes[i]
        weights[class .& fav] .= n₊*sum(class)/(n*sum(class .& fav))
        weights[class .& unfav] .= (n-n₊)*sum(class)/(n*sum(class .& unfav))
    end
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
end

"""
    ReweighingWrapper(classifier=nothing, grp=:class)

Instantiates a ReweighingWrapper which wrapper the `classifier` with the Reweighing fairness algorithm.
The sensitive attribute can be specified by the parameter `grp`.
If `classifier` doesn't support weights while training, an error is thrown.
"""
function ReweighingWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class)
    model = ReweighingWrapper(grp, classifier)
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

    grps = X[:, model.grp]

    weights = _calculateWeights(grps, y)

    Xs = source(X)
    ys = source(y)
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
end

"""
    ReweighingSamplingWrapper(classifier=nothing, grp=:class, factor=1, rng=Random.GLOBAL_RNG)

Instantiates a ReweighingSamplingWrapper which wrapper the classifier with the Reweighing fairness algorithm together with sampling.
The sensitive attribute can be specified by the parameter `grp`.
`factor`*number_of_samples_in_original_data datapoints are sampled using calculated weights and then used to train after sampling from the reweighed dataset.
A negative or no value value for `factor` parameter instructs the algorithm to use the same number of datapoints as in original sample.
"""
function ReweighingSamplingWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class, factor::Float64=1.0, rng=nothing)
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if rng == nothing
        rng = Random.GLOBAL_RNG
    end
    model = ReweighingSamplingWrapper(grp, classifier, factor, rng)
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
    grps = X[:, model.grp]
    n = model.factor<0 ? length(y) : Int(round(model.factor*length(y)))
    weights = _calculateWeights(grps, y)
    indices = sample(model.rng, 1:length(y), FrequencyWeights(weights), n; replace=true, ordered=false)
    Xnew = X[indices, :]
    ynew = y[indices]
    Xs = source(Xnew)
    ys = source(ynew)
    mach1 = machine(model.classifier, Xs, ys)
    ŷ = MMI.predict(mach1, Xs)

    mach2 = machine(Deterministic(), Xs, ys; predict=ŷ)
    return!(mach2, model, verbosity)
end

MMI.input_scitype(::Type{<:ReweighingSamplingWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:ReweighingSamplingWrapper{M}}) where M = AbstractVector{<:Finite{2}}
