const FairTensor = AxisArray

"""
    FairTensor(m; labels, outcomes=[1, 0])

Uses labels and array m to create FairTensor. Returns Fairness Tensor which consists of C 2 x 2 matrices stacked up to form a Matrix
of size C x 2 x 2 where C is the number of groups. Each 2 x 2 matrix contains values [[TP, FP], [FN, TN]].
The FairTensor returned is of type AxisArray and has both, group labels and values.
"""
function FairnessTensor(m::Array{R, 3}; labels::Vector{String}, outcomes::Vector=[0, 1]) where R
    s = size(m)
    (s[3] == s[2] && s[3] == 2) || throw(ArgumentError("Expected a C*2*2 type Matrix."))
    length(labels) == s[1] ||
        throw(ArgumentError("As many labels as classes must be provided."))
    return AxisArray(m, labels=labels, pred=outcomes, truth=outcomes)
end

function Base.getproperty(obj::AxisArray, sym::Symbol)
   if sym === :labels
       return obj[Axis{:labels}].val
   elseif sym === :mat
       return obj.data
   else # fallback to getfield
       return getfield(obj, sym)
   end
end

"""
    fair_tensor(ŷ, y, grp)

Computes the fairness tensor, where ŷ are the predicted classes,
y are the ground truth values, grp are the group values.
The ordering follows that of `levels(y)`.

Note that ŷ, y and grp are all categorical arrays
"""
function fair_tensor(ŷ::Vec{<:CategoricalElement}, y::Vec{<:CategoricalElement},
                          grp::Vec{<:CategoricalElement})

    check_dimensions(ŷ, y)
    check_dimensions(ŷ, grp)
    length(levels(y))==2 || throw(ArgumentError("Binary Targets are only supported"))
    outcomes = levels(y)
    favLabel = outcomes[2]
    unfavLabel = outcomes[1]

    levels_ = levels(grp)
    c = length(levels_)
    # Dictionary data-structure is used now to map group labels and the corresponding index.
    # Other alternative could be binary search on levels_ everytime. But it would be slow by log(length(levels_)).
    grp_idx = Dict()
    for i in 1:c
        grp_idx[levels_[i]] = i
    end

    # Coverting Categorical Vector to Bool Vector.
    # TODO: Can think of adding another dispatch where user directly passes Bool Vec
    y = y.==favLabel
    ŷ = ŷ.==favLabel
    n = length(y)

    fact = zeros(Int, c, 2, 2)
    @inbounds for i in 1:n
        if ŷ[i] && y[i]
            fact[grp_idx[grp[i]], 1, 1] += 1
        elseif ŷ[i] && !y[i]
            fact[grp_idx[grp[i]], 1, 2] += 1
        elseif !ŷ[i] && y[i]
            fact[grp_idx[grp[i]], 2, 1] += 1
        elseif !y[i] && !y[i]
            fact[grp_idx[grp[i]], 2, 2] += 1
        end
    end
    return AxisArray(fact, labels=string.(levels_), pred=[unfavLabel, favLabel], truth=[unfavLabel, favLabel])
end

# synonym
fact = fair_tensor
