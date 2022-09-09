# Fairness.jl

[![Build Status](https://github.com/ashryaagr/Fairness.jl/workflows/CI/badge.svg)](https://github.com/ashryaagr/Fairness.jl/actions)
[![Build status](https://ci.appveyor.com/api/projects/status/lsh7co54fplsdl4q?svg=true)](https://ci.appveyor.com/project/ashryaagr/fairness-jl)
[![Coverage Status](https://codecov.io/gh/ashryaagr/Fairness.jl/branch/master/graph/badge.svg?token=wbrk8MSeMp)](https://codecov.io/gh/ashryaagr/Fairness.jl)
<a href="https://slackinvite.julialang.org/">
  <img src="https://img.shields.io/badge/chat-on%20slack-orange.svg"
       alt="#mlj">
</a>
<a href="https://www.ashrya.in/Fairness.jl/dev/">
  <img src="https://img.shields.io/badge/docs-stable-blue.svg"
       alt="Documentation">
</a>

[Fairness.jl](https://github.com/ashryaagr/Fairness.jl) is a comprehensive bias audit and mitigation toolkit in julia. Extensive support and functionality provided by [MLJ](https://github.com/alan-turing-institute/MLJ.jl) has been used in this package.

For an introduction to Fairness.jl refer the notebook available at https://nextjournal.com/ashryaagr/fairness

# Installation
```julia
using Pkg
Pkg.activate("my_environment", shared=true)
Pkg.add("Fairness")
Pkg.add("MLJ")
```

# What Fairness.jl offers over its alternatives?
- As of writing, it is the only bias audit and mitigation toolkit to support data with multi-valued protected attribute. For eg. If the protected attribute, say race has more than 2 values: "Asian", "African", "American"..so on, then Fairness.jl can easily handle it with normal workflow.
- Multiple Fairness algorithms can be applied at the same time by wrapping the wrapped Model. [Example is available in Documentation](https://www.ashrya.in/Fairness.jl/dev/algorithms/#Composability)
- Due to the support for multi-valued protected attribute, intersectional fairness can also be dealt with this toolkit. For eg. If the data has 2 protected attributes, say race and gender, then Fairness.jl can be used to handle it by combining the attributes like "female_american", "male_asian"...so on.
- Extensive support and functionality provided by [MLJ](https://github.com/alan-turing-institute/MLJ.jl) can be leveraged when using Fairness.jl
- Tuning of models using MLJTuning from MLJ. Numerious ML models from MLJModels can be used together with Fairness.jl
- It leverages the flexibility and speed of Julia to make it more efficient and easy-to-use at the same time
- Well structured and intutive design
- Extensive tests and Documentation

# Getting Started

- [Documentation](https://www.ashrya.in/Fairness.jl/dev) is a good starting point for this package.
- To understand Fairness.jl, it is recommended that the user goes through the [MLJ Documentation](https://alan-turing-institute.github.io/MLJ.jl/stable/). It shall help the user in understanding the usage of machine, evaluate, etc.
- Incase of any difficulty or confusion feel free to [open an issue](https://github.com/ashryaagr/Fairness.jl/issues/new).

# Example
Following is an introductory example of using Fairness.jl. Observe how easy it has become to measure and mitigate bias in Machine Learning algorithms.
```julia
using Fairness, MLJ
X, y, ŷ = @load_toydata

julia> model = ConstantClassifier()
ConstantClassifier() @904

julia> wrappedModel = ReweighingSamplingWrapper(classifier=model, grp=:Sex)
ReweighingSamplingWrapper(
    grp = :Sex,
    classifier = ConstantClassifier(),
    factor = 1) @312

julia> evaluate(
          wrappedModel,
          X, y,
          measures=MetricWrappers(
              [true_positive, true_positive_rate], grp=:Sex))
┌────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────── ⋯
│ _.measure          │ _.measurement                                                                       │ _.per_fold                           ⋯
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────── ⋯
│ true_positive      │ Dict{Any,Any}("M" => 2,"overall" => 4,"F" => 2)                                     │ Dict{Any,Any}[Dict("M" => 0,"overall ⋯
│ true_positive_rate │ Dict{Any,Any}("M" => 0.8333333333333334,"overall" => 0.8333333333333334,"F" => 1.0) │ Dict{Any,Any}[Dict("M" => 4.99999999 ⋯
└────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────── ⋯
```

# Components
Fairness.jl is divided into following components

### FairTensor
It is a 3D matrix of values of TruePositives, False Negatives, etc for each group. It greatly helps in optimization and removing the redundant calculations.

### Measures

#### CalcMetrics

| Name | Metric Instances |
|-----|-------------|
| True Positive | truepositive,  true_positive
| True Negative | truenegative, true_negative
| False Positive | falsepositive, false_positive
| False Negative | falsenegative, false_negative
| True Positive Rate | truepositive_rate, true_positive_rate, tpr, recall, sensitivity, hit_rate
| True Negative Rate | truenegative_rate, true_negative_rate, tnr, specificity, selectivity
| False Positive Rate | falsepositive_rate, false_positive_rate, fpr, fallout
| False Negative Rate | falsenegative_rate, false_negative_rate, fnr, miss_rate
| False Discovery Rate | falsediscovery_rate, false_discovery_rate, fdr
| Precision | positivepredictive_value, positive_predictive_value, ppv
| Negative Predictive Value | negativepredictive_value, negative_predictive_value, npv

#### FairMetrics

| Name | Formula | Value for Custom function (func)
|-----|-------------|----------------|
| disparity | metric(Gᵢ)/metric(RefGrp) ∀ i| func(metric(Gᵢ), metric(RefGrp)) ∀ i
| parity | [ (1-ϵ) <= dispariy_value[i] <= 1/(1-ϵ) ∀ i ] | [ func(disparity_value[i]) ∀ i ]

#### BoolMetrics [WIP]
These metrics shall use either parity or shall have custom implementation to return boolean values

| Metric | Aliases |
|-----|-------------|
| Demographic Parity | DemographicParity

### Fairness Algorithms
These algorithms are wrappers. These help in mitigating bias and improve fairness.

| Algorithm Name | Metric Optimised | Supports Multi-valued protected attribute | Type | Reference |
|----------------|------------------|-------------------------------------------|------|-----------|
| Reweighing | General | :heavy_check_mark: |  Preprocessing | [Kamiran and Calders, 2012](http://doi.org/10.1007/s10115-011-0463-8)
| Reweighing-Sampling | General | :heavy_check_mark: | Preprocessing | [Kamiran and Calders, 2012](http://doi.org/10.1007/s10115-011-0463-8)
| Equalized Odds Algorithm | Equalized Odds | :heavy_check_mark: | Postprocessing | [Hardt et al., 2016](https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning)
| Calibrated Equalized Odds Algorithm | Calibrated Equalized Odds | :x: | Postprocessing | [Pleiss et al., 2017](https://proceedings.neurips.cc/paper/2017/file/b8b9c74ac526fffbeb2d39ab038d1cd7-Paper.pdf)
| LinProg Algorithm | Any metric | :heavy_check_mark: | Postprocessing | Our own Algorithm
| Penalty Algorithm | Any metric | :heavy_check_mark: | Inprocessing | Our own Algorithm

# Contributing

- Various Contribution opportunities are available. Some of the possible contributions have been listed at [the pinned issue](https://github.com/ashryaagr/Fairness.jl/issues/3#issuecomment-656812338)
- Feel free to open an issue or contact on slack. Let us know where your intersts and strengths lie and we can find possible contribution opportunities for you.

# Citing Fairness.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3977197.svg)](https://doi.org/10.5281/zenodo.3977197)

```bibtex
@software{ashrya_agrawal_2020_3977197,
  author       = {Ashrya Agrawal and
                  Jiahao Chen and
                  Sebastian Vollmer and
                  Anthony Blaom},
  title        = {ashryaagr/Fairness.jl},
  month        = aug,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.1.2},
  doi          = {10.5281/zenodo.3977197},
  url          = {https://doi.org/10.5281/zenodo.3977197}
}
```
