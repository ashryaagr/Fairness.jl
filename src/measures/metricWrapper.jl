struct Mean <: MLJBase.AggregationMode end

function (::Mean)(dicts::Array{Dict{Any, Any}})
    valuesDict = Dict()
    grps = keys(dicts[1])
    n = length(dicts)
    for grp in grps
        valSum = 0
        for dict in dicts
            valSum += dict[grp]
        end
        valuesDict[grp] = valSum/n
    end
    return valuesDict
end

struct MetricWrapper <: MLJBase.Measure
    measure::MLJBase.Measure
    grp::Symbol
end

function MetricWrapper(measure::MLJBase.Measure; grp=:class)
    MetricWrapper(measure, grp)
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
MLJBase.target_scitype(::Type{<:MetricWrapper}) = AbstractArray{Multiclass{2},1}
MLJBase.supports_weights(::Type{<:MetricWrapper}) = false # for now
# MLJBase.prediction_type(::Type{<:MetricWrapper}) = :deterministic # Not specifying it to have check_measures false
MLJBase.orientation(::Type{<:MetricWrapper}) = :other # other options are :score, :loss
MLJBase.reports_each_observation(::Type{<:MetricWrapper}) = false
MLJBase.aggregation(::Type{<:MetricWrapper}) = Mean()
MLJBase.is_feature_dependent(::Type{<:MetricWrapper}) = true
