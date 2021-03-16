@testset "Equalized Odds Postprocessing" begin
	X, y, _ = @load_toydata;
	model = ConstantClassifier()
	wrappedModel = EqOddsWrapper(classifier=model, grp=:Sex)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[6:10, :])
	@test all(ŷ .== 1)
	@test length(ŷ)==5
	# TODO:test the fitresult of machine
end


@testset "Calibrated Equalized Odds Postprocessing" begin
	X, y, _ = @load_toydata;
	model = ConstantClassifier()
	wrappedModel = CalEqOddsWrapper(classifier=model, grp=:Sex)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[6:10, :])
	@test all(ŷ .== 1)
	@test length(ŷ)==5
	# TODO:test the fitresult of machine
end

@testset "LinProgWrapper Postprocessing" begin
	X, y, _ = @load_toydata;
	model = ConstantClassifier()
	wrappedModel = LinProgWrapper(classifier=model, grp=:Sex, measure=true_positive_rate)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[6:10, :])
	@test length(ŷ)==5
	# TODO:test the fitresult of machine
end
