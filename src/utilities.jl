"""
    check_dimension(X, Y)

Check that two vectors or matrices have matching dimensions
"""
function check_dimensions(X::AbstractVecOrMat, Y::AbstractVecOrMat)
    size(X, 1) == size(Y, 1) ||
        throw(DimensionMismatch("The two objects don't have the same " *
                                "number of rows."))
    return nothing
end
