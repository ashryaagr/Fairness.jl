mutable struct DemographicParity <: Measure
    C::Int
    A::Array # Adp Matrix/array which is on the LHS of equation
    # This matrix A has been kept in the struct for efficiency in case of multiple calls on a single dataset
    # Add a constructor for zero matrix here
    DemographicParity() = new(0, [])
end


function (dp::DemographicParity)(ft::FairTensor{C}) where C
    if dp.C==0
        # Here matrix A of DemographicParity will be initialised as it wasn't previously initialised
        dp.C = C
        dp.A = zeros(Int, C, 8)
        Nc = sum(ft[C, :, :])
        for i in 1:C-1
            Ni = sum(ft[i, :, :])
            dp.A[i, 1] = Ni
            dp.A[i, 3] = Ni
            dp.A[i, 5] = -Nc
            dp.A[i, 7] = -Nc
        end
    end
    z = reshape(ft.mat, C, 4)
    z_c = reshape(repeat(z[C, :], C), 4, C)
    z = hcat(z, transpose(z_c))
    return all(dp.A*transpose(z) .== 0)
end
