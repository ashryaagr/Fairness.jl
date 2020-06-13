"
    _ftIdx(ft, grp)

Finds the index of grp (string) in ft.labels which corresponds to ft.mat.
For Index i for the grp  returned by this function ft[i, :, :] returns the
2D array [[TP, FP], [FN, TN]] for that group.
"
function _ftIdx(ft::FairTensor, grp)
    idx = findfirst(x->x==string.(grp), ft.labels)
    if idx==nothing throw(ArgumentError("$grp not found in the fairness tensor")) end
    return idx
end

# Helper function for calculating TruePositive, FalseNegative, etc.
# If grp is :, it calculates combined value for all groups, else the specified group
_calcmetric(ft::FairTensor, grp, inds...) = typeof(grp)==Colon ? sum(ft.mat[:, inds...]) : sum(ft.mat[_ftIdx(ft, grp), inds...])

(::TruePositive)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 1, 1)
(::FalsePositive)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 1, 2)
(::FalseNegative)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 2, 1)
(::TrueNegative)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 2, 2)


# The functions true_positive, false_negative, etc are instances of TruePositive, FalseNegative, etc.
# So on using true_positive(fair_tensor) will use the above defined functions
(::TPR)(ft::FairTensor; grp=:) = 1/(1+false_negative(ft; grp=grp)/true_positive(ft; grp=grp))
(::TNR)(ft::FairTensor; grp=:) = 1/(1+false_positive(ft; grp=grp)/true_negative(ft; grp=grp))
(::FPR)(ft::FairTensor; grp=:) = 1-true_negative_rate(ft; grp=grp)
(::FNR)(ft::FairTensor; grp=:) = 1-true_positive_rate(ft; grp=grp)


(::FDR)(ft::FairTensor; grp=:) = 1/(1+false_positive(ft; grp=grp)/true_positive(ft; grp=grp))
(::Precision)(ft::FairTensor; grp=:) = 1 - false_discovery_rate(ft; grp=grp)
(::NPV)(ft::FairTensor; grp=:) =  1/(1+false_negative(ft; grp=grp)/true_negative(ft; grp=grp))


"""
    disparity(M, ft; refGrp=nothing, func=/)

Computes disparity for fairness tensor `ft` with respect to an array of metrics `M`.

For any class A and a reference Group B, `disparity = func(metric(A), metric(B))`.
By default `func` is `/` .

A dataframe is returned with disparity values for all combinations of metrics and classes.
It contains a column named labels for the classes and has a column for disparity of each metric in M.
The column names are metric names appended with `_disparity`.

## Keywords

* `refGroup=nothing` : The reference group
* `func=/` : The function used to evaluate disparity. This function should take 2 arguments.
The second argument shall correspond to reference group.

Please note that division by 0 will result in NaN
"""
function disparity(M::Vector{<:Measure}, ft::FairTensor{C}; refGrp=nothing, func=/) where C
    refGrp!==nothing || throw(ArgumentError("Value of reference group needs to be provideds"))
    refGrpIdx = _ftIdx(ft, refGrp)
    df = DataFrame(labels=ft.labels)
    for m in M
        colName = string(m) * "_disparity"
        colDisparity = Symbol(colName)
        # TODO : _ftIdx is repeatedly called internally. Instead the value can be stored and reused.
        baseVal = m(ft; grp=refGrp)
        arr = zeros(Float64, C)
        for i in 1:C
            arr[i] = func(m(ft; grp=ft.labels[i]), baseVal)
        end
        df[:, colDisparity] = arr
    end
    return df
end
