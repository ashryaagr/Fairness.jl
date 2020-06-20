module MLJFair

# ===================================================================
# IMPORTS
using Tables
using MLJBase
using CategoricalArrays
using DataFrames
using MLJModels, MLJModelInterface
using StatsBase
# ===================================================================
## METHOD EXPORTS

export fair_tensor, fact
export _ftIdx

# Export the Boolean Metrics
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
       falsediscovery_rate, false_discovery_rate, fdr, npv, ppv,
       recall, sensitivity, hit_rate, miss_rate,
       specificity, selectivity, f1score, fallout

# Export the fairness metrics
export disparity, parity

export ReweighingWrapper, ReweighingSamplingWrapper

# -------------------------------------------------------------------
# re-export From CategoricalArrays
export categorical, levels, levels!

# re-export from MLJBase
export pretty
# ===================================================================
## CONSTANTS

const CategoricalElement = Union{CategoricalValue,CategoricalString}
const Vec = AbstractVector
const Measure =  MLJBase.Measure
const MMI =  MLJModelInterface

# ===================================================================
# Includes

include("utilities.jl")
include("fair_tensor.jl")
include("measures/measures.jl")
include("algorithms/algorithms.jl")

end # module
