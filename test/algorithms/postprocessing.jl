@testset "Equalized Odds Postprocessing" begin
	X, y = @load_compas;
	indices = X.race.!=levels(X.race)[5] #This group is only 0.2%
	X, y = X[indices, :], y[indices]
	X.race = categorical(convert(Array, X.race))
	model = ConstantClassifier()
	wrappedModel = EqOddsWrapper(classifier=model, grp=:race)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[6:10, :])
	@test length(ŷ)==5
	# TODO:test the fitresult of machine
end

@testset "LinProgWrapper Postprocessing" begin
	X, y= @load_compas;
	indices = X.race.!=levels(X.race)[5] #This group is only 0.2%
	X, y = X[indices, :], y[indices]
	X.race = categorical(convert(Array, X.race))
	model = ConstantClassifier()
	wrappedModel = LinProgWrapper(classifier=model, grp=:race, measure=true_positive_rate)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[6:10, :])
	@test length(ŷ)==5
	# TODO:test the fitresult of machine
end
