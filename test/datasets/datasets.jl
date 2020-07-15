@testset "COMPAS Dataset" begin
	X, y = @load_compas
	fpath = joinpath(MLJFair.DATA_DIR, "compas-scores-two-years.csv")
	@test isfile(fpath)
	# @test scitype(y) == AbstractArray{Multiclass{2}, 1}
	cols = ["sex", "age", "age_cat", "race", "c_charge_degree", "priors_count", "days_b_screening_arrest", "decile_score"]
	@test string.(names(X)) == cols
end

@testset "Adult Dataset" begin
	X, y = @load_adult
	fpath = joinpath(MLJFair.DATA_DIR, "adult.data")
	@test isfile(fpath)
	@test scitype(y) == AbstractArray{Multiclass{2}, 1}
	cols = ["age", "workclass", "fnlwgt", "education",
		"education_num", "marital_status", "occupation",
		"relationship", "race", "sex", "capital_gain",
		"capital_loss", "hours_per_week", "native_country",
	]
	@test string.(names(X)) == cols
end

@testset "German Dataset" begin
	X, y = @load_german
	fpath = joinpath(MLJFair.DATA_DIR, "german.data")
	@test isfile(fpath)
	@test scitype(y) == AbstractArray{Multiclass{2},1}
	cols = [
		"status", "duration", "credit_history",
		"purpose", "credit_amount", "savings", "employment",
		"installment_rate", "gender_status",
		"other_debtors", "residence_since", "property", "age",
		"installment_plans", "housing", "existing_credits",
		"skill_level", "people_liable", "telephone",
		"foreign_worker"
	]
	@test string.(names(X)) == cols
end
