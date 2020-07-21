@testset "Reweighing" begin
	X, y, _ = @load_toydata;
	model = ConstantClassifier()
	wrappedModel = ReweighingWrapper(classifier=model, grp=:Sex)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[1:5, :])
	@test all(x-> typeof(x) <: UnivariateFinite, ŷ)
	@test length(ŷ)==5
end

@testset "ReweighingSampling" begin
	X, y, _ = @load_toydata;
	model = ConstantClassifier()
	wrappedModel = ReweighingSamplingWrapper(classifier=model, grp=:Sex)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[1:5, :])
	@test all(x-> typeof(x) <: UnivariateFinite, ŷ)
	@test length(ŷ)==5
end
