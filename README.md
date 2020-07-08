# MLJFair

[MLJFair](https://github.com/ashryaagr/MLJFair.jl) is a bias audit and mitigation toolkit in julia and is supported by MLJ Ecosystem.
It is being developed as a part of JSOC 2020 Program sponsored by JuliaComputing.

# Installation
```julia
using Pkg
Pkg.activate("my_environment", shared=true)
Pkg.add("https://github.com/ashryaagr/MLJFair.jl")
Pkg.add("MLJ")
```

# What MLJFair offers over its alternatives?
- As of writing, it is the only bias audit and mitigation toolkit to support data with multi-valued protected attribute. For eg. If the protected attribute, say race has more than 2 values: "Asian", "African", "American"..so on, then MLJFair can easily handle it with normal workflow.
- Due to the support for multi-valued protected attribute, intersectional fairness can also be dealt with this toolkit. For eg. If the data has 2 protected attributes, say race and gender, then MLJFair can be used to handle it by combining the attributes like "female_american", "male_asian"...so on.
- Extensive support and functionality provided by [MLJ](https://github.com/alan-turing-institute/MLJ.jl) can be leveraged when using MLJFair.
- Tuning of models using MLJTuning from MLJ. Numerious ML models from MLJModels can be used together with MLJFair.
- It leverages the flexibility and speed of Julia to make it more efficient and easy-to-use at the same time
- Well structured and intutive design
- Extensive tests and Documentation

# Example
Following is an introductory example of using MLJFair. Observe how easy it has become to measure and mitigate bias in Machine Learning algorithms.
```julia
using MLJFair, MLJ
X, y, ŷ = @load_toydata

julia> model = ConstantClassifier()
ConstantClassifier() @904

julia> wrappedModel = ReweighingSamplingWrapper(model, grp=:Sex)
ReweighingSamplingWrapper(
    grp = :Sex,
    classifier = ConstantClassifier(),
    noSamples = -1) @312

julia> evaluate(
          wrappedModel,
          X, y,
          measure=MetricWrapper(
              true_positive_rate; grp=:Sex))

Evaluating over 6 folds: 100%[=========================] Time: 0:00:12
┌───────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────── ⋯
│ _.measure     │ _.measurement                                                                                      │ _.per_fold                                     ⋯
├───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────── ⋯
│ MetricWrapper │ Dict{Any,Any}("M" => 0.6666666666666669,"overall" => 0.5000000000000003,"F" => 0.8333333333333335) │ Dict{Any,Any}[Dict("M" => 4.999999999999998e-1 ⋯
└───────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────── ⋯
```

# Components
MLJFair is divided into following components

### FairTensor
It is a 3D matrix of values of TruePositives, False Negatives, etc for each group. It greatly helps in optimization and removing the redundant calculations.

### Measures
  - CalcMetrics
    - true_positive_rate, false_positive....
  - FairMetrics
    - disparity
    - parity
  - BoolMetrics
    - DemographicParity
  - MetricWrapper

### Fairness Algorithms
These algorithms are wrappers. These help in mitigating bias and improve fairness.
  - Preprocessing Algorithms
      - Reweighing
      - ReweighingSampling
  - PostProcessing Algorithms
      - Equalized Odds PostProcessing
      - LinProg PostProcessing (Generalizes the Equalized Odds algorithm for any metric)
  - InProcessing Algorithms
      - Meta-Fair algorithm with provable guarantees[WIP]

# Getting Started
- [Examples and tutorials](https://github.com/ashryaagr/MLJFair.jl/tree/master/examples) are a good starting point.
- [Documentation](https://www.ashrya.in/MLJFair.jl/dev) is also available for this package.
