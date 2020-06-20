"""
Helper function for ReweighingWrapper.
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

mutable struct ReweighingWrapper <: DeterministicNetwork
    grp::Symbol
    classifier::MLJBase.Model
end

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

Weights for each group-label combination is calculated.
Using the calculated weights, rows are sampled uniformly. The weight is used to sample uniformly.
"""
mutable struct ReweighingSamplingWrapper <: DeterministicNetwork
    grp::Symbol
    classifier::MLJBase.Model
end

function ReweighingSamplingWrapper(classifier::MLJBase.Model; grp::Symbol=:class)
    return ReweighingSamplingWrapper(grp, classifier)
end

function MLJBase.fit(model::ReweighingSamplingWrapper,
    verbosity::Int, X, y)
    grps = X[:, model.grp]
    n = length(y)
    weights = _calculateWeights(grps, y)
    indices = sample(1:n, FrequencyWeights(weights), n; replace=true, ordered=false)
    Xnew = X[indices, :]
    ynew = y[indices]
    Xs = source(Xnew, kind=:input)
    ys = source(ynew, kind=:target)
    mch = machine(model.classifier, Xs, ys)
    ẑ = MMI.predict(mch, Xs)
    ŷ = ẑ
    fit!(ŷ, verbosity=0)
    return fitresults(ŷ)
end
