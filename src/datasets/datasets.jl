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
        Downloads.download(url, file)
    end
end

"
Macro to read csv file of job data (data/jobs.csv) and convert columns to categorical.
Returns the tuple (X, y, ŷ)
"
macro load_toydata()
    quote
        fpath = joinpath(DATA_DIR, "jobs.csv")
        data = DataFrame(CSV.File(fpath); copycols = false)
        transform!(data, names(data)[1:6] .=> categorical, renamecols=false)
        (data[!, names(data)[1:4]], data[!, :Class], data[!, :Pred])
    end
end

"
Macro to create fairness Tensor for data/jobs.csv
The fairness tensor will be created on the basis of the column Job Type.
This column has 3 different values for job types.
"
macro load_toyfairtensor()
    quote
        X, y, ŷ = @load_toydata
        fair_tensor(ŷ, y, X[!, names(X)[4]])
    end
end

"""
Macro to load [COMPAS dataset](https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb)
It is a reduced version of COMPAS Datset with 8 features and 6907 rows. The protected attributes are sex and race.
The available features are used to predict whether a criminal defendant will recidivate(reoffend).

Returns (X, y)
"""
macro load_compas()
    quote
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        fname = "compas-scores-two-years.csv"
        ensure_download(url, fname)

        fpath = joinpath(DATA_DIR, fname)
        data = DataFrame(CSV.File(fpath); copycols = false)
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

"""
Macro to Load the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)
It has 14 features and 32561 rows. The protected attributes are race and sex.
This dataset is used to predict whether income exceeds 50K dollars per year.

Returns (X, y)
"""
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
        data = DataFrame(CSV.File(fpath, header=cols, silencewarnings=true, delim=", "); copycols = false)
        # Warning is silenced to supress warnings for lesser number of columns

        data = dropmissing(data, names(data))
        data.income_per_year = map(data.income_per_year) do η
            η == "<=50K" ? 0 : 1
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
        df = DataFrame(CSV.File(fpath, header=cols); copycols = false)

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

"""
The data is related with direct marketing campaigns of a Portuguese banking institution.
The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required,
in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
It has 20 features and 41188 rows. The protected attributes is marital.
"""
macro load_bank_marketing()
    quote
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
        fname = "bank-marketing.csv"
        fpath = joinpath(DATA_DIR, fname)
        if !isfile(fpath)
            Downloads.download(url, "tempdataset.zip")
            zarchive = ZipFile.Reader("tempdataset.zip")
            zipfile = filter(x->x.name=="bank-additional/bank-additional-full.csv", zarchive.files)[1]
            df = DataFrame(CSV.File(read(zipfile)); copycols = false)
            CSV.write(fpath, df)
            close(zarchive)
            Base.Filesystem.rm("tempdataset.zip", recursive=true)
        end
        df = DataFrame(CSV.File(fpath); copycols = false)
        coerce!(df, Textual => Multiclass)
        coerce(df, :y => Multiclass)
        (y, X) = unpack(df, ==(:y), col->true)
        y = categorical(y)
        levels!(y, ["yes", "no"])
        (X, y)
    end
end

# TODO: Construct protected attribute from the attributes: blackpct, whitepct, asianpct
"""
The per capita violent crimes variable was calculated using population and
the sum of crime variables considered violent crimes in the United States: murder, rape, robbery, and assault.
It has 127 features and 1994 rows. The protected attributes are ....?
"""
macro load_communities_crime()
    quote
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
        fname = "communities_crime.data"
        ensure_download(url, fname)
        fpath = joinpath(DATA_DIR, fname)
        df = DataFrame(CSV.File(fname, header=false); copycols = false)
        df = dropmissing(df, names(df))
        X = df[!, names(df)[1:127]]
        y = df[!, names(df)[128]] .> 0.7
        y = categorical(y)
        (X, y)
    end
end

"""
Student Performance Dataset. It has 395 rows and 30 features.
The target attribute corresponds to grade G1. The target tells whether the student gets grade >= 12.
The protected attribute is sex.
"""
macro load_student_performance()
    quote
        url= "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        fname = "student-performance.csv"
        fpath = joinpath(DATA_DIR, fname)
        if !isfile(fpath)
            Downloads.download(url, "tempdataset.zip")
            zarchive = ZipFile.Reader("tempdataset.zip")
            zipfile = filter(x->x.name=="student-mat.csv", zarchive.files)[1]
            df = DataFrame(CSV.File(read(zipfile)); copycols = false)
            CSV.write(fpath, df)
            close(zarchive)
            Base.Filesystem.rm("tempdataset.zip", recursive=true)
        end
        df = DataFrame(CSV.File(fpath); copycols = false)
        coerce!(df, Textual => Multiclass)
        X = df[!, names(df)[1:30]]
        y = df[!, names(df)[31]] .>= 12
        y = categorical(y)
        (X, y)
    end
end
