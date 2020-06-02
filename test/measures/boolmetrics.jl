@testset "Boolean Fairness Metrics" begin
    ft = job_fairtensor()
    dp = DemographicParity()
    @test dp(ft) == false
    @test dp.C == 3
    A = zeros(3, 8)
    A[1, [1, 3, 5, 7]] = [4, 4, -3, -3]
    A[2, [1, 3, 5, 7]] = [3, 3, -3, -3]
    @test dp.A == A
end
