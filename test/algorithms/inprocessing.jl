@testset "PenaltyWrapper Inprocessing"
	X, y = @load_compas;
	model = @pipeline ContinuousEncoder @load(RandomForestClassifier, pkg=DecisionTree)
	wrappedModel = PenaltyWrapper(classifier=model, grp=:sex, measure=tpr)
	mach = machine(wrappedModel, X, y)
	@test_broken length(predict(fit!(mach), X)) == length(y)
	# TODO:test the fitresult of machines
end
