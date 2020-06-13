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
