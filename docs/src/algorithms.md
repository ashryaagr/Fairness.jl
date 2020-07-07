# Fairness Algorithms
MLJFair provides with various algorithms that can help in mitigating bias and improving fairness metrics.

## Introduction
These algorithms are wrappers.
As demonstrated in last section, these wrappers can be used to compose a complex pipeline with more than 1 fairness algorithm. These fairness algorithms have been divided into 3 categories based on the parts in the pipeline that the algorithm can control. These 3 categories are Preprocessing, Postprocessing and Inprocessing[WIP].

## Preprocessing Algorithms
These are the algorithms that have control over the training data to be fed into machine learning model.
This class of algorithms improves the representation of groups in the training data.

### ReweighingSampling Algorithm
This algorithm also supports training data where more than 2 values are possible for sensitive attribute.
```@docs
ReweighingSamplingWrapper
MLJFair.ReweighingSamplingWrapper(::MLJBase.Model)
```

### Reweighing Algorithm
This model being wrapped with this wrapper needs to support weights. If the model doesn't support training using weights, then error is thrown. In case weights are not supported by your desired model, them switch to ReweighingSampling Algorithm.
To find the models in MLJ that support weights, execute:
```julia
using MLJ
for model in models()
   if model.supports_weights println(model.name) end
end
```
```@docs
ReweighingWrapper
MLJFair.ReweighingWrapper(::MLJBase.Model)
MLJFair._calculateWeights
```

## Postprocessing
These are the algorithms that have control over the final predictions. They can tweak final predictions to optimise fairness constraints.

### Equalized Odds Algorithm
This algorithm supports training data with 2 or more possible values for the sensitive attribute.
```@docs
EqOddsWrapper
MLJFair.EqOddsWrapper(::MLJBase.Model)
```

### LinProg Algorithm
This algorithm supports all the metrics provided by MLJFair. It hasn't been tested on other possible user defined metrics. It supports training data with binary sensitive attribute.
```@docs
LinProgWrapper
MLJFair.LinProgWrapper(::MLJBase.Model)
```

## Composability

MLJFair provides you the ability to easily use multiple fairness algorithms on top of each other.
A fairness algorithm can be added over another fairness algorithm by simply wrapping the previous wrapped model with the new wrapper. MLJFair handles everything else for you!
The use of wrappers provides you the ability to add as many algorithms as you want!!

```@repl
using MLJFair, MLJBase, MLJModels
X, y, _ = @load_toydata;
model = ConstantClassifier();
wrappedModel = ReweighingSamplingWrapper(model; grp=:Sex);
wrappedModel2 = EqOddsWrapper(wrappedModel; grp=:Sex);
mch = machine(wrappedModel2, X, y);
fit!(mch)
ŷ = predict(mch, X);
```
