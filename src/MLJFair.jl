module MLJFair

# ===================================================================
# IMPORTS
using Tables
using MLJBase
using CategoricalArrays

# ===================================================================
## METHOD EXPORTS

export fair_tensor, fact

# ===================================================================
## CONSTANTS

const CategoricalElement = Union{CategoricalValue,CategoricalString}
const Vec = AbstractVector

# ===================================================================
# Includes

include("utilities.jl")
include("fair_tensor.jl")

end # module
