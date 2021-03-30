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

function models_fn(fp::FairnessProblem, alpha=1.0)
	########## You can change code segment below to add more models

	rf = @pipeline ContinuousEncoder ConstantClassifier

	models = MLJBase.Model[rf]
	#  Just simply add the base classifiers above and to this list

	model_names = ["RF"] # Add names of models. Preferably keep names short

	################################################################
	model_count = length(models)
	protected_attr = fp.task.grp
	debiasmeasures = fp.task.debiasmeasures
	for i in 1:model_count
    rw = ReweighingSamplingWrapper(classifier=models[i], grp=protected_attr, alpha=alpha)
		lp = LinProgWrapper(classifier=models[i], grp=protected_attr,
								measures=debiasmeasures, alpha=alpha)
		eo = EqOddsWrapper(classifier=models[i], grp=protected_attr, alpha=alpha)
        ce = CalEqOddsWrapper(classifier=models[i], grp=protected_attr,fp_rate=0, fn_rate=1, alpha=alpha)
		append!(models, [rw, lp, eo, ce])
		append!(model_names, model_names[i]*"-".*["Reweighing",
		"LinProg-".*join(debiasmeasures, "-"), "Equalized Odds","Calibrated Equalized Odds"])
	end
	return models, model_names
end

@testset "Get df" begin
	@test size(get_df(student(), models_fn))==(48, 7)
end

@testset "Get Pareto df" begin
	@test size(get_pareto_df(student(), models_fn, 0:1:0.5))==(48, 8)
end
