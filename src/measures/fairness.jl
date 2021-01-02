"""
    disparity(M, ft; refGrp=nothing, func=/)

Computes disparity for fairness tensor `ft` with respect to an array of metrics `M`.

For any class A and a reference Group B, `disparity = func(metric(A), metric(B))`.
By default `func` is `/` .

A dataframe is returned with disparity values for all combinations of metrics and classes.
It contains a column named labels for the classes and has a column for disparity of each metric in M.
The column names are metric names appended with `_disparity`.

## Keywords

* `refGrp=nothing` : The reference group
* `func=/` : The function used to evaluate disparity. This function should take 2 arguments.
The second argument shall correspond to reference group.

Please note that division by 0 will result in NaN
"""
function disparity(measures::Vector, ft::FairTensor{C}; refGrp=nothing, func=/) where C
    refGrp!==nothing || throw(ArgumentError("Value of reference group needs to be provided"))
    refGrpIdx = _ftIdx(ft, refGrp)
    df = DataFrame(labels=ft.labels)
    for measure in measures
        colName = MMI.name(measure) * "_disparity"
        colDisparity = Symbol(colName)
        # TODO : _ftIdx is repeatedly called internally. Instead the value can be stored and reused.
        baseVal = measure(ft, grp=refGrp)
        arr = zeros(Float64, C)
        for i in 1:C
            arr[i] = func(measure(ft, grp=ft.labels[i]), baseVal)
        end
        df[:, colDisparity] = arr
    end
    return df
end

"""
    Disparity

Disparity uses the `disparity` function and has information about the protected attribute, reference Group and custom function
It will help in automatic evaluation using MLJ.evaluate
"""
struct Disparity <: MLJBase.Measure
    measure::Measure
    grp::Symbol
    refGrp
    func
end

"""
    Disparity(measure, grp=:class, refGrp=nothing, func=/)

Instantiates the struct Disparity.
"""
function Disparity(measure::Measure; grp::Symbol=:class, refGrp=nothing, func=/)
    Disparity(measure, grp, refGrp, func)
end

"""
    Disparities(measures, grp=:class, refGrp=nothing, func=/)

Creates instances of Disparity struct for each measure in measures.
"""
function Disparities(measures::Vector{<:Measure}; grp::Symbol, refGrp=nothing, func=/)
    wrappedMetrics = []
    for measure in measures
        push!(wrappedMetrics, Disparity(measure; grp=grp, refGrp=refGrp, func=func))
    end
    return wrappedMetrics
end

function (D::Disparity)(ŷ, X, y)
    if typeof(ŷ) <: UnivariateFiniteArray
        ŷ = StatsBase.mode.(ŷ)
    end
    if typeof(y) <: UnivariateFiniteArray
        y = StatsBase.mode.(y)
    end
    grps = X[:, D.grp]
    n_grps = length(levels(grps))

    ft = fair_tensor(categorical(ŷ), categorical(y), categorical(grps))

    valuesDict = Dict()
    df = disparity([D.measure], ft,refGrp=D.refGrp, func=D.func)
    vals = df[!, names(df)[2]]
    for i in 1:n_grps
        valuesDict[df.labels[i]]=vals[i]
    end

    valuesDict["overall"] = D.func(D.measure(ft), D.measure(ft, grp=D.refGrp))
    return valuesDict
end

MLJBase.name(::Type{<:Disparity}) = "Disparity"
# MLJBase.target_scitype(::Type{<:Disparity}) = AbstractArray{Multiclass{2},1}
MLJBase.supports_weights(::Type{<:Disparity}) = false # for now
# MLJBase.prediction_type(::Type{<:Disparity}) = :deterministic # Not specifying it to have check_measures false
MLJBase.orientation(::Type{<:Disparity}) = :other # other options are :score, :loss
MLJBase.reports_each_observation(::Type{<:Disparity}) = false
MLJBase.aggregation(::Type{<:Disparity}) = Mean()
MLJBase.is_feature_dependent(::Type{<:Disparity}) = true

# To display the name of actual measure while printing instead of the name Disparity
Base.show(stream::IO, D::Disparity) = print(stream, string(D.measure)*"_disparity")
#------------------------------------------------------------------
# Helper for parity function. Default criteria to calculate parity
_parity_func(ϵ) = x -> isnan(x) ? NaN : (1-ϵ) <= x <= 1/(1-ϵ)

"""
    parity(df, ϵ=nothing; func=nothing)

Takes the dataframe `df` returned from `disparity` function and adds columns for parity values corresponding to each disparity column.
It then calculates the parity values for measures that were passed for disparity calculation.

Parity is a boolean value indicating whether a fairness constraint is satisfied by disparity values.

The default fairness criteria for a disparity value `x` and fairness threshold `ϵ` for a group is:

    (1-ϵ) <= x <= 1/(1-ϵ)

Here ϵ is the fairness threshold which is required if default fairness constraint is used.

## Keywords

* `func=nothing` : single argument function specifying custom Fairness constraint for disparity instead of the default criteria
"""
function parity(df::DataFrame, ϵ=nothing; func=nothing)
    if func==nothing
        ϵ!=nothing || throw(ArgumentError("For using default fairness criteria, ϵ i.e. fairness threshold should be passed."))
        func = _parity_func(ϵ)
    end
    cols = names(df)
    M = string.(cols)
    for i in 1:length(cols)
        if !endswith(M[i], "_disparity")
            continue
        end
        m = replace(M[i], "disparity" => "parity")
        m = Symbol(m)
        df[:, m] = func.(df[:, cols[i]])
    end
    return df
end
