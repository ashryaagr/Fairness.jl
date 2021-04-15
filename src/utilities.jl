# Check that two vectors or matrices have matching dimensions
function check_dimensions(X::AbstractVecOrMat, Y::AbstractVecOrMat)
    size(X, 1) == size(Y, 1) ||
        throw(DimensionMismatch("The two objects don't have the same " *
                                "number of rows."))
    return nothing
end


function show(ft::Fairness.FairTensor)
    for i=1:length(ft.labels)
        show(ft.labels[i])
        show("\n")
        show(ft[i,:,:])
    end
end