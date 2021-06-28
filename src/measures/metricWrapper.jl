struct Mean <: MLJBase.StatisticalTraits.AggregationMode end

function (::Mean)(dicts)
    valuesDict = Dict()
    grps = keys(dicts[1])
    n = length(dicts)
    for grp in grps
        valSum = 0
        for dict in dicts
            valSum += dict[grp]
        end
        if typeof(valSum)==Int64
            valuesDict[grp] = valSum # When the individual metric value is an integer(eg. for true_positive), then aggregation is via sum
        else
            valuesDict[grp] = valSum/n # When the metric is a float, then it is a rate(eg. for true_positive_rate). So, aggregation is via mean
        end
    end
    return valuesDict
end

function (::Mean)(dict::Dict{Any,Any})
    return dict
end

"""
    MetricWrapper

MetricWrapper wraps the fairness metrics and has the information about the protected attribute.

"""
mutable struct MetricWrapper <: MLJBase.Measure
    measure::MLJBase.Measure
    grp::Symbol
end
# This struct is made mutable to avoid repetitive specification of grp attribute in functions like fairevaluate

"""
    MetricWrapper(measure, grp=:class)

Instantiates the struct MetricWrapper.
"""
function MetricWrapper(measure::MLJBase.Measure; grp=:class)
    MetricWrapper(measure, grp)
end

"""
    MetricWrappers(measures, grp=:class)

Creates MetricWrapper for multiple metrics at same time.
"""
function MetricWrappers(measures; grp=:class)
    wrappedMetrics = []
    for measure in measures
        push!(wrappedMetrics, MetricWrapper(measure; grp=grp))
    end
    return wrappedMetrics
end

function (FM::MetricWrapper)(ŷ, X, y)
    if typeof(ŷ) <: UnivariateFiniteArray
        ŷ = StatsBase.mode.(ŷ)
    end
    if typeof(y) <: UnivariateFiniteArray
        y = StatsBase.mode.(y)
    end
    grps = X[:, FM.grp]
    ft = fair_tensor(categorical(ŷ), categorical(y), categorical(grps))

    valuesDict = Dict()
    valuesDict["overall"] = FM.measure(ft)
    for grp in levels(grps)
        valuesDict[grp] = FM.measure(ft; grp=grp)
    end
    return valuesDict
end


MLJBase.name(::Type{<:MetricWrapper}) = "MetricWrapper"
# MLJBase.target_scitype(::Type{<:MetricWrapper}) = AbstractArray{Multiclass{2},1}
MLJBase.supports_weights(::Type{<:MetricWrapper}) = false # for now
# MLJBase.prediction_type(::Type{<:MetricWrapper}) = :deterministic # Not specifying it to have check_measures false
MLJBase.orientation(::Type{<:MetricWrapper}) = :other # other options are :score, :loss
MLJBase.reports_each_observation(::Type{<:MetricWrapper}) = false
MLJBase.aggregation(::Type{<:MetricWrapper}) = Mean()
MLJBase.is_feature_dependent(::Type{<:MetricWrapper}) = true

# To display the name of actual measure while printing instead of the name MetricWrapper
Base.show(stream::IO, m::MetricWrapper) = print(stream, m.measure)
