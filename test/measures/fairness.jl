@testset "disparity" begin
    ft = @load_toyfairtensor
    M = [true_positive_rate, false_positive_rate, ppv]
    @test_throws ArgumentError disparity(M, ft)
    d = disparity(M, ft; refGrp="Education")
    @test names(d) == ["labels", "true_positive_rate_disparity", "false_positive_rate_disparity",
                        "positive_predictive_value_disparity"]
    @test size(d)==(3, 4)
end

@testset "Disparity" begin
    X, y = @load_toydata()
    model = ConstantClassifier()
    result = evaluate(model, X, y, measure=Disparity(tpr, grp=:Sex, refGrp="M", func=-))
    @test length(keys(result.per_fold[1][1])) == 3
    result = evaluate(model, X, y, measures=Disparities([tpr, fpr], grp=:Sex, refGrp="M", func=/))
    @test length(keys(result.per_fold[1][1])) == 3
    @test length(result.per_fold) == 2
end

@testset "Parity" begin
    ft = @load_toyfairtensor
    M = [true_positive_rate]
    df = disparity(M, ft; refGrp="Board")
    df = parity(df, 0.2)
    @test names(df)[3] == "true_positive_rate_parity"
    arr = df[:, :true_positive_rate_parity]
    @test arr[1] && !arr[2] && !arr[3]
end
