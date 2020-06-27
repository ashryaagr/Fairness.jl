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
mutable struct ReweighingWrapper <: DeterministicNetwork
    grp::Symbol
    classifier::MLJBase.Model
end

"""
    ReweighingWrapper(classifier; grp=:class)

Instantiates a ReweighingWrapper which wrapper the `classifier` with the Reweighing fairness algorithm.
The sensitive attribute can be specified by the parameter `grp`.
If `classifier` doesn't support weights while training, an error is thrown.
"""
function ReweighingWrapper(classifier::MLJBase.Model; grp::Symbol=:class)
    supports_weights(classifier) || throw(ArgumentError("Classifier provided does not support weights"))
    return ReweighingWrapper(grp, classifier)
end

function MLJBase.fit(model::ReweighingWrapper,
    verbosity::Int, X, y)

    Xs = source(X)
    ys = source(y, kind=:target)

    stand_model = Standardizer()
    stand = machine(stand_model, Xs)
    W = transform(stand, Xs)

    grps = X[:, model.grp]

    weights = _calculateWeights(grps, y)

    classifier = model.classifier
    cls_mch = machine(classifier, W, ys, weights)
    ŷ= MMI.predict(cls_mch, W)

    fit!(ŷ, verbosity=0)

    return fitresults(ŷ)
end


"""
    ReweighingSamplingWrapper

ReweighingSamplingWrapper is a preprocessing algorithm wrapper in which Weights for each group-label combination is calculated.
Using the calculated weights, rows are sampled uniformly. The weight is used to sample uniformly.
The number of datapoints used to train after sampling from the reweighed dataset can be specified by `noSamples`.
"""
mutable struct ReweighingSamplingWrapper <: DeterministicNetwork
    grp::Symbol
    classifier::MLJBase.Model
    noSamples::Int
end

"""
    ReweighingSamplingWrapper(classifier; grp=:class, noSamples=-1)

Instantiates a ReweighingSamplingWrapper which wrapper the classifier with the Reweighing fairness algorithm together with sampling.
The sensitive attribute can be specified by the parameter `grp`.
`noSamples` indicates the number of datapoints used to train after sampling from the reweighed dataset.
A negative or no value value for `noSamples` parameter instructs the algorithm to use the same number of datapoints as in original sample.
"""
function ReweighingSamplingWrapper(classifier::MLJBase.Model; grp::Symbol=:class, noSamples::Int=-1)
    return ReweighingSamplingWrapper(grp, classifier, noSamples)
end

function MLJBase.fit(model::ReweighingSamplingWrapper,
    verbosity::Int, X, y)
    grps = X[:, model.grp]
    n = model.noSamples<0 ? length(y) : model.noSamples
    weights = _calculateWeights(grps, y)
    indices = sample(1:length(y), FrequencyWeights(weights), n; replace=true, ordered=false)
    Xnew = X[indices, :]
    ynew = y[indices]
    Xs = source(Xnew, kind=:input)
    ys = source(ynew, kind=:target)
    mch = machine(model.classifier, Xs, ys)
    ŷ = MMI.predict(mch, Xs)
    fit!(ŷ, verbosity=0)
    return fitresults(ŷ)
end
