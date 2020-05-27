include(joinpath("data", "data.jl"))

@testset "fair_tensor" begin
    ft = job_fairtensor()
    @test ft.mat == cat([2 2; 0 0; 0 2], [0 0; 2 1; 1 0], dims=3)
    @test Set(ft.labels) == Set(["Board", "Education", "Healthcare"])
end
