"""

"""
struct FairTensor{C}
    mat::Matrix
    labels::Vector{String}
end

"""

"""
function FairTensor(m::Matrix{Int}, labels::Vector{String})
end

# allow to access ft[i,j] but not set (it's immutable)
Base.getindex(ft::FairTensor, inds...) = getindex(ft.mat, inds...)

"""

"""
function fair_tensor(ŷ::Vec{<:CategoricalElement}, y::Vec{<:CategoricalElement};
                          rev::Union{Nothing,Bool}=nothing,
                          perm::Union{Nothing,Vector{<:Integer}}=nothing,
                          warn::Bool=true)
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
