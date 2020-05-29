"""
    FairTensor{C}

Fairness Tensor with C classes. It consists of C 2*2 matrices stacked up to form a Matrix
of size 2*2*C. Each 2*2 matrix contains values [[TP, FP], [FN, TN]].
"""
struct FairTensor{C}
    mat::Array{Int, 3}
    labels::Vector{String}
end

"""
    FairTensor(m, labels)

Instantiates a FairTensor using the matrix m.
"""
function FairTensor(m::Array{Int, 3}, labels::Vector{String})
    s = size(m)
    (s[3] == s[2] && s[3] == 2) || throw(ArgumentError("Expected a 2*2*C type Matrix."))
    length(labels) == s[1] ||
        throw(ArgumentError("As many labels as classes must be provided."))
    FairTensor{s[1]}(m, labels)
end

# allow to access ft[i,j] but not set (it's immutable)
Base.getindex(ft::FairTensor, inds...) = getindex(ft.mat, inds...)

"""
    fair_tensor(ŷ, y, grp)

Computes the fairness tensor, where ŷ are the predicted classes,
y are the ground truth values, grp are the group values.
The ordering follows that of `levels(y)`.

"""
function fair_tensor(ŷ::Vec{<:CategoricalElement}, y::Vec{<:CategoricalElement},
                          grp::Vec{<:CategoricalElement})

    check_dimensions(ŷ, y)
    Set(levels(y))==Set([0, 1]) || throw(ArgumentError("Values in y should be binary in nature."))
    Set(levels(ŷ))==Set([0, 1]) || throw(ArgumentError("Values in ŷ should be binary in nature."))
    check_dimensions(ŷ, grp)
    # TODO: Handle the case with levels(y) = 0 or 1 or ϕ specially. Maybe keep an optional argument for the label

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
    y = convert(AbstractVector{Bool}, y)
    ŷ = convert(AbstractVector{Bool}, ŷ)
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
    return FairTensor(fact, string.(levels_))

end

# synonym
fact = fair_tensor

# aggregation:
Base.round(m::FairTensor; kws...) = m
function Base.:+(t1::FairTensor, t2::FairTensor)
    if t1.labels != t2.labels
        throw(ArgumentError("Tensor labels must agree"))
    end
    FairTensor(t1.mat + t2.mat, t1.labels)
end
