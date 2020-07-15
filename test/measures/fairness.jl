@testset "Parity" begin
    ft = @load_toyfairtensor
    M = [true_positive_rate]
    df = disparity(m, ft; refGrp="Board")
    df = parity(df, 0.2)
    @test names(df)[3] == :true_positive_rate_parity
    arr = df[:, :true_positive_rate_parity]
    @test arr[1] && isnan(arr[2]) && !arr[3]
end
