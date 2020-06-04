module MLJFair

# ===================================================================
# IMPORTS
using Tables
using MLJBase
using CategoricalArrays
using DataFrames

# ===================================================================
## METHOD EXPORTS

export fair_tensor, fact
export DemographicParity

#Export the metric instances from MLJBase to permit calculation of metrics without using MLJBase
export TruePositive, TrueNegative, FalsePositive, FalseNegative,
       TruePositiveRate, TrueNegativeRate, FalsePositiveRate,
       FalseNegativeRate, FalseDiscoveryRate, Precision, NPV,
       # standard synonyms
       TPR, TNR, FPR, FNR, FDR, PPV,
       # instances and their synonyms
       truepositive, truenegative, falsepositive, falsenegative,
       true_positive, true_negative, false_positive, false_negative,
       truepositive_rate, truenegative_rate, falsepositive_rate,
       true_positive_rate, true_negative_rate, false_positive_rate,
       falsenegative_rate, negativepredictive_value,
       false_negative_rate, negative_predictive_value,
       positivepredictive_value, positive_predictive_value,
       tpr, tnr, fpr, fnr,
       falsediscovery_rate, false_discovery_rate, fdr, npv, ppv

export disparity

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
