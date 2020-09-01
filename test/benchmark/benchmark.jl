@testset "FairEvaluate Benchmarking" begin
	model = ConstantClassifier()
	X, y = @load_toydata
	results = fairevaluate([model, model], X, y, grp=:Sex, priv_grps=["M"])
	@test length(results)==2
	@test size(results[1])==(2, 2)
	@test size(results[2])==(2, 2)
end
