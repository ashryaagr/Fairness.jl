module MLJFair

# ===================================================================
# IMPORTS
using Tables
using MLJBase
using CategoricalArray

# ===================================================================
## METHOD EXPORTS

export fair_tensor, fact

# ===================================================================
# Includes

include("fair_tensor.jl")

end # module
