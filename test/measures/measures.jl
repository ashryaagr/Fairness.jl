module TestMeasures

using MLJFair, Test

include(joinpath("..", "data", "data.jl"))

include("calcmetrics.jl")
include("boolmetrics.jl")

end  # module
