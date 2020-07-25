@testset "Constructors of Basic Fairness Calc. Metrics" begin
    ft = @load_toyfairtensor
    # check all constructors
    # These exend the struct from MLJ Base https://github.com/ashryaagr/Fairness.jl/blob/1d0093232ff215ea8a7e8521b0612162f70a92c3/src/measures/calcmetrics.jl#L18
    m = TruePositive()
    @test m(ft) == truepositive(ft)
    m = TrueNegative()
    @test m(ft) == truenegative(ft)
    m = FalsePositive()
    @test m(ft) == falsepositive(ft)
    m = FalseNegative()
    @test m(ft) == falsenegative(ft)
    m = TruePositiveRate()
    @test m(ft) == tpr(ft) == truepositive_rate(ft)
    m = TrueNegativeRate()
    @test m(ft) == tnr(ft) == truenegative_rate(ft)
    m = FalsePositiveRate()
    @test m(ft) == fpr(ft) == falsepositive_rate(ft)
    m = FalseNegativeRate()
    @test m(ft) == fnr(ft) == falsenegative_rate(ft)
    m = FalseDiscoveryRate()
    @test m(ft) == fdr(ft) == falsediscovery_rate(ft)
    m = Precision()
    @test m(ft) == positive_predictive_value(ft)
    m = NPV()
    @test m(ft) == npv(ft)
    # check synonyms
    m = TPR()
    @test m(ft) == tpr(ft)
    m = TNR()
    @test m(ft) == tnr(ft)
    m = FPR()
    @test m(ft) == fpr(ft) == fallout(ft)
    m = FNR()
    @test m(ft) == fnr(ft) == miss_rate(ft)
    m = FDR()
    @test m(ft) == fdr(ft)
end

@testset "Values for Basic Fairness Calc. Metrics" begin
    ft = @load_toyfairtensor
    @test true_positive(ft) == 2
    @test truenegative(ft) == 1
    @test falsepositive(ft) == 3
    @test falsenegative(ft) == 4
    @test truepositive_rate(ft) ≈ 1/3
    @test truenegative_rate(ft) ≈ 0.25
    @test falsenegative_rate(ft) ≈ 2/3
    @test falsepositive_rate(ft) ≈ 0.75
    @test falsediscovery_rate(ft) ≈ 0.4
    @test positive_predictive_value(ft) ≈ 0.6
    @test negative_predictive_value(ft) ≈ 0.2
    @test accuracy(ft) == 0.3
end

@testset "Group Specific Calc. Metrics" begin
    ft = @load_toyfairtensor
    @test Fairness._ftIdx(ft, "Education") == 2
    @test_throws ArgumentError Fairness._ftIdx(ft, "ABCDE")
    @test true_positive(ft; grp=ft.labels[1]) == 2
    @test truenegative(ft; grp=ft.labels[2]) == 1
    @test falsepositive(ft; grp=ft.labels[3]) == 1
    @test falsenegative(ft; grp=ft.labels[1]) == 2
    @test truepositive_rate(ft; grp=ft.labels[1]) ≈ 0.5
    @test truenegative_rate(ft; grp=ft.labels[2]) ≈ 1/3
    @test falsepositive_rate(ft; grp=ft.labels[2]) ≈ 2/3
    @test isapprox(falsenegative_rate(ft; grp=ft.labels[2]), 0.0; atol=1e-10, rtol=0)
    @test isapprox(falsediscovery_rate(ft; grp=ft.labels[3]), 0.0; atol=1e-10, rtol=0)
    @test positive_predictive_value(ft; grp=ft.labels[3]) ≈ 1.0
    @test isapprox(negative_predictive_value(ft; grp=ft.labels[3]) , 0.0; atol=1e-10, rtol=0)
    @test accuracy(ft; grp=ft.labels[1]) == 0.5
end
