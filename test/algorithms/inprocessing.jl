@testset "PenaltyWrapper Inprocessing" begin
	X, y = @load_compas;
	model = @pipeline ContinuousEncoder @load(RandomForestClassifier, pkg=DecisionTree)
	wrappedModel = PenaltyWrapper(classifier=model, grp=:sex, measure=tpr, n_iters=10)
	mach = machine(wrappedModel, X, y)
	@test length(predict(fit!(mach), X)) == length(y)
	# TODO:test the fitresult of machines
end
