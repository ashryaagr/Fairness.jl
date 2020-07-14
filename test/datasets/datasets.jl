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
