@testset "FairEvaluate Benchmarking" begin
	model = ConstantClassifier()
	X, y = @load_toydata
	results = fairevaluate([model, model], X, y, grp=:Sex, priv_grps=["M"])
	@test length(results)==5
	@test size(results["measures"])==(1,)
	@test size(results["pvalues"])==(1, 2, 2)
	@test size(results["classifier_names"])==(2,)
	@test size(results["results"])==(1, 2, 6)
	@test size(results["tstats"])==(1, 2, 2)
end
