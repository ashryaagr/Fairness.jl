module TestFairTensor

using Test
using Fairness

@testset "fair_tensor" begin
    ft = @load_toyfairtensor
    @test ft.mat == cat([2 2; 0 0; 0 2], [0 0; 2 1; 1 0], dims=3)
    @test Set(ft.labels) == Set(["Board", "Education", "Healthcare"])

    # Fairness Tensor Addition
    ft_1 = ft+ft
    @test ft_1.mat == ft.mat+ft.mat
end

end  # module
