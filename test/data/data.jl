using CSV
using MLJFair
using DataFrames

"
Utility function to read csv file of job data and convert columns to categorical
"
function jobdata()
    fpath = joinpath(@__DIR__, "..", "..", "data", "jobs.csv")
    data = DataFrame!(CSV.File(fpath))
    categorical!(data, names(data)[1:6])
    return data
end

"
Utility function for tests to create fairness Tensor for jobs.csv
The fairness tensor will be created on the basis of the column Job Type.
This column has 3 different values for job types
"
function job_fairtensor()
    data = jobdata()
    return fair_tensor(data[!, :Pred], data[!, :Class], data[!, names(data)[4]])
end
