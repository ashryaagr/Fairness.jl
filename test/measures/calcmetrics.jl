@testset "Constructors of Basic Fairness Calc. Metrics" begin
    ft = job_fairtensor()
    # check all constructors
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
    ft = job_fairtensor()
    @test true_positive(ft) == 2
    @test truenegative(ft) == 1
    @test falsepositive(ft) == 3
    @test falsenegative(ft) == 4
    @test truepositive_rate(ft) ≈ 1/3
    @test truenegative_rate(ft) == 0.25
    @test falsepositive_rate(ft) == 0.75
    @test falsenegative_rate(ft) ≈ 2/3
    @test falsediscovery_rate(ft) == 0.4
    @test positive_predictive_value(ft) == 0.6
    @test negative_predictive_value(ft) == 0.2
end
