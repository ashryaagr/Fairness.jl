using MLJFair
using MLJBase, MLJModels
using Test

@testset "fair_tensor" begin
    include("fair_tensor.jl")
end

@testset "measures" begin
    include("measures/measures.jl")
end

@testset "algorithms" begin
    include("algorithms/algorithms.jl")

end
