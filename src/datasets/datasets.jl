const DATA_DIR = joinpath(MODULE_DIR, "..", "data")

const COERCE_ADULT = (
    :age => Continuous,
    :workclass => Multiclass,
    :fnlwgt => Continuous,
    :education => Multiclass,
    :education_num => Continuous,
    :marital_status => Multiclass,
    :occupation => Multiclass,
    :relationship => Multiclass,
    :race => Multiclass,
    :sex => Multiclass,
    :capital_gain => Continuous,
    :capital_loss => Continuous,
    :hours_per_week => Continuous,
    :native_country => Multiclass,
    :income_per_year => Multiclass,
)

"""
Checks whether the dataset is already present in data directory. Downloads it if not present.
"""
function ensure_download(url::String, file::String)
    cd(DATA_DIR) # This is to ensue that the dataset is not downloaded to /tmp instead of ./data
    fpath = joinpath(DATA_DIR, file)
    if !isfile(fpath)
        download(url, file)
    end
end

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

"
Macro to load COMPAS dataset.
https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
"
macro load_compas()
    quote
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        fname = "compas-scores-two-years.csv"
        ensure_download(url, fname)

        fpath = joinpath(DATA_DIR, fname)
        data = DataFrame!(CSV.File(fpath))
        data = data[!, ["sex", "age", "age_cat", "race", "c_charge_degree", "priors_count", "days_b_screening_arrest", "decile_score", "is_recid"]]
        dropmissing!(data, disallowmissing=true)
        coerce!(data, Textual => Multiclass)
        coerce!(data, :is_recid => Multiclass)
        y, X = unpack(data, ==(:is_recid), col -> true)

        X = data[!, ["sex", "age", "age_cat", "race", "c_charge_degree", "priors_count", "days_b_screening_arrest", "decile_score"]]
        y = data[!, "is_recid"]
        (X, y)
    end
end

"Macro to load Adult dataset."
macro load_adult()
    quote
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        fname = "adult.data"
        cols = ["age", "workclass", "fnlwgt", "education",
            "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain",
            "capital_loss", "hours_per_week", "native_country",
            "income_per_year"
        ]
        ensure_download(url, fname)
        fpath = joinpath(DATA_DIR, fname)
        data = DataFrame!(CSV.File(fpath, header=cols))

        data = dropmissing(data, names(data))
        data.income_per_year = map(data.income_per_year) do η
            η == " <=50K" ? 0 : 1
        end

        coerce!(data, COERCE_ADULT...)
        coerce!(data, :income_per_year => Multiclass)
        y, X = unpack(data, ==(:income_per_year), col -> true)
        (X, y)
    end
end

"""
Load the full version of [German credit dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29).
This dataset has 20 features and 1000 rows. The protected attributes are gender_status and age (>25 is priviledged)
Using the 20 features, it classifies the credit decision to a person as good or bad credit risks.

Returns (X, y)
"""
macro load_german()
    quote
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        fname = "german.data"
        cols = [
            "status", "duration", "credit_history",
            "purpose", "credit_amount", "savings", "employment",
            "installment_rate", "gender_status",
            "other_debtors", "residence_since", "property", "age",
            "installment_plans", "housing", "existing_credits",
            "skill_level", "people_liable", "telephone",
            "foreign_worker", "label"
        ]
        ensure_download(url, fname)
        fpath = joinpath(DATA_DIR, fname)
        df = DataFrame!(CSV.File(fpath, header=cols))

        gender_status = Dict(
            "A91" => "male_divorced_separated",
            "A92" => "female_divorced_separated_married",
            "A93" => "male_single",
            "A94" => "male_married_widowed",
            "A95" => "female_single", # There seems to be no row with this value in data
        )

        df.gender_status = map(df.gender_status) do η
            gender_status[η]
        end

        coerce!(df, Textual => Multiclass)
        coerce!(df, :label => Multiclass)
        y, X = unpack(df, ==(:label), col -> true);
        (X, y)
    end
end
