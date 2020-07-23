@testset "MetaFair Inprocessing" begin
	X, y, _ = @load_toydata;
	model = ConstantClassifier()
	wrappedModel = MetaFairWrapper(classifier=model, grp=:Sex, measure=tpr)
	mach = machine(wrappedModel, X, y)
	fit!(mach)
	@test_broken length(predict(mach, X[6:10, :]))== 5
end
