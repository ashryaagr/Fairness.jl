module TestMeasures

using Fairness, Test
using MLJBase, MLJModels

include("calcmetrics.jl")
include("boolmetrics.jl")
include("fairness.jl")
include("metricWrapper.jl")

end  # module
