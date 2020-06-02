module TestMeasures

using MLJFair, Test
using MLJBase

include(joinpath("..", "data", "data.jl"))

include("calcmetrics.jl")
include("boolmetrics.jl")

end  # module
