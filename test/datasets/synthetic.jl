@testset "ZafarData" begin
	X, y = genZafarData()
	@test scitype(y) == AbstractArray{Multiclass{2}, 1}
	@test string.(names(X)) == ["x1", "x2", "z"]
end

@testset "ZafarData2" begin
  X, y = genZafarData2()
  @test scitype(y) == AbstractArray{Multiclass{2}, 1}
  @test string.(names(X)) == ["x1", "x2", "z"]
end

@testset "SubgroupData" begin
	X, y = genSubgroupData()
	@test scitype(y) == AbstractArray{Multiclass{2}, 1}
	@test string.(names(X)) == ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "z"]
end

@testset "BiasedSampleData" begin
  X, y = genBiasedSampleData()
  @test scitype(y) == AbstractArray{Multiclass{2}, 1}
  @test string.(names(X)) == ["x1", "x2", "z"]
end
