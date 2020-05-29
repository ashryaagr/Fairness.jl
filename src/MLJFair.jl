module MLJFair

# ===================================================================
# IMPORTS
using Tables
using MLJBase
using CategoricalArrays

# ===================================================================
## METHOD EXPORTS

export fair_tensor, fact
export fairZ
export DemographicParity

# ===================================================================
## CONSTANTS

const CategoricalElement = Union{CategoricalValue,CategoricalString}
const Vec = AbstractVector
const Measure =  MLJBase.Measure

# ===================================================================
# Includes

include("utilities.jl")
include("fair_tensor.jl")
include("measures/measures.jl")

end # module
