@testset "ReweighingSampling" begin
	X, y, _ = @load_toydata;
	model = ConstantClassifier()
	wrappedModel = ReweighingSamplingWrapper(model; grp=:Sex)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	ŷ = predict(mach, X[1:5, :])
	@test all(x-> UnivariateFinite(categorical([0, 1]), [0.5, 0.5]).prob_given_class == x.prob_given_class, ŷ)
	@test length(ŷ)==5
end
