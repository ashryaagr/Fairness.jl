@testset "Metric Wrapper" begin
	X, y = @load_toydata
	model = ConstantClassifier()
	result = evaluate(model, X, y, measure=MetricWrapper(tpr, grp=:Sex))
	@test length(keys(result.per_fold[1][1])) == 3
	result = evaluate(model, X, y, measures=[cross_entropy, MetricWrappers([tpr, fnr], grp=:Sex)...])
	@test length(keys(result.per_fold[2][1])) == 3
	@test length(result.per_fold) == 3
end
