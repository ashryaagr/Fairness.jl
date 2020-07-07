const DATA_DIR = joinpath(MODULE_DIR, "..", "data")

"""
    load_dataset(fpath, coercions)

Load one of standard dataset like Boston etc assuming the file is a
comma separated file with a header.

"""
function load_dataset(fname::String, coercions::Tuple)
    fpath = joinpath(DATA_DIR, fname)
    data_raw, data_header = readdlm(fpath, ',', header=true)
    data_table = Tables.table(data_raw; header=Symbol.(vec(data_header)))
    return coerce(data_table, coercions...; tight=true)
end
# The above function is taken from MLJBase/src/data/datasets.jl
# TODO: Use Above function to load various datasets like COMPAS, German, etc.

"
Macro to read csv file of job data and convert columns to categorical.
Returns the tuple (X, y, ŷ)
"
macro load_toydata()
    quote
        fpath = joinpath(DATA_DIR, "jobs.csv")
        data = DataFrame!(CSV.File(fpath))
        categorical!(data, names(data)[1:6])
        (data[!, names(data)[1:4]], data[!, :Class], data[!, :Pred])
    end
end

"
Macro to create fairness Tensor for jobs.csv
The fairness tensor will be created on the basis of the column Job Type.
This column has 3 different values for job types.
"
macro load_toyfairtensor()
    quote
        X, y, ŷ = @load_toydata
        fair_tensor(ŷ, y, X[!, names(X)[4]])
    end
end
