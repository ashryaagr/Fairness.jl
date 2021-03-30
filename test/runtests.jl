using Fairness
using MLJBase, MLJModels
using Test

@testset "fair_tensor" begin
    include("fair_tensor.jl")
end

@testset "measures" begin
    include("measures/measures.jl")
end

@testset "dataset_macros" begin
    include("datasets/datasets.jl")
    include("datasets/synthetic.jl")
end

@testset "algorithms" begin
    include("algorithms/algorithms.jl")
end

@testset "benchmarking" begin
    include("benchmark/benchmark.jl")
end
