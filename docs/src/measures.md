# Measures

In MLJFair, measures are callable structs. Their instances are also available to be called directly.

The instances can be called by passing fairness tensor to it. Its general form is metric(ft::FairTensor; grp=nothing). The instances have multiple aliases for convinience.

The Measures have been divided into calcMetrics and boolmetrics depending on whether the metric returns Numerical value or Boolean value respectively.

MLJFair._ftIdx is a utility function that has been used to calculate metrics and shall be helpful while using MLJFair to inspect Fairness tensor values.
```@docs
MLJFair._ftIdx
```

## CalcMetrics

These metrics return a Numerical Value.

### CalcMetrics - Usage
These measures all have the common calling syntax

```julia
measure(ft)
```

or

```julia
measure(ft; grp)
```
where ft is the fairness tensor. Here `grp` is an optional, named, string parameter used to compute the fairness metric for a specific group. If `grp` is not specified, the overall value of fairness metric is calculated.

```@repl measures
using MLJFair
ŷ = categorical([1, 0, 1, 1, 0]);
y = categorical([0, 0, 1, 1, 1]);
grp = categorical(["Asian", "African", "Asian", "American", "African"]);
ft = fair_tensor(ŷ, y, grp);
TruePositiveRate()(ft)
true_positive_rate(ft) # true_positive_rate is instance of TruePositiveRate
true_positive_rate(ft; grp="Asian")
```

##### Following Metrics (callable structs) are available through MLJFair :

`TruePositive`, `TrueNegative`, `FalsePositive`, `FalseNegative`,
`TruePositiveRate`, `TrueNegativeRate`, `FalsePositiveRate`,
`FalseNegativeRate`, `FalseDiscoveryRate`, `Precision`, `NPV`

#### standard synonyms of above Metrics
`TPR`, `TNR`, `FPR`, `FNR`, `FDR`, `PPV`,

#### instances of above metrics and their synonyms
`truepositive`, `truenegative`, `falsepositive`, `falsenegative`,
`true_positive`, `true_negative`, `false_positive`, `false_negative`,
`truepositive_rate`, `truenegative_rate`, `falsepositive_rate`,
`true_positive_rate`, `true_negative_rate`, `false_positive_rate`,
`falsenegative_rate`, `negativepredictive_value`,
`false_negative_rate`, `negative_predictive_value`,
`positivepredictive_value`, `positive_predictive_value`,
`tpr`, `tnr`, `fpr`, `fnr`,
`falsediscovery_rate`, `false_discovery_rate`, `fdr`, `npv`, `ppv`,
`recall`, `sensitivity`, `hit_rate`, `miss_rate`,
`specificity`, `selectivity`, `fallout`

###
```@docs
disparity
```

```@repl measures
M = [true_positive_rate, positive_predictive_value];
disparity(M, ft; refGrp="Asian")
```

## BoolMetrics

These metrics return a boolean value.
These metrics are callable structs. The struct has field for the A and C. A corresponds to the matrix on LHS of the equality-check equation A*z = 0 in [this paper's](https://arxiv.org/pdf/2004.03424.pdf), Equation No. 3. In this paper it is a 1D array. But to deal with multiple group fairness, a 2D array matrix is used.

Initially the instatiated metric contains 0 and [] as values for C and A. But after calling it on fairness tensor, the values of C and A change as shown below. This gives the advantage to reuse the same instantiation again. But upon reusing, the matrix A need not be generated again as it will remain the same. This makes it faster!

### DemographicParity

```@repl measures
dp = DemographicParity()
dp.A, dp.C # Initial values in struct DemographicParity
dp(ft)
dp.A, dp.C # New values in dp (instance of DemographicParity)
```
