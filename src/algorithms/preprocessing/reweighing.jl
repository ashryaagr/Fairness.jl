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

    grps = X[:, model.grp]
    classes = levels(grps)
    n_classes = length(classes)
    n = length(y)

    fav = y.==levels(y)[2] # Boolean
    unfav = y.==levels(y)[1]
    n₊ = sum(fav) # Total Number of favourable outcomes

    weights = zeros(Float64, n)
    for i in 1:n_classes
        class = grps.==classes[i]
        weights[class .& fav] .= n₊*sum(class)/(n*sum(class .& fav))
        weights[class .& unfav] .= (n-n₊)*sum(class)/(n*sum(class .& unfav))
    end

    classifier = model.classifier
    cls_mch = machine(classifier, X, y)
    fit!(cls_mch, verbosity=0)
    return fitresults(cls_mch)
end
