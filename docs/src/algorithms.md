# Fairness Algorithms
MLJFair provides with various algorithms that can help in mitigating bias and improving fairness metrics.

## Introduction
These algorithms are wrappers.
As demonstrated in last section, these wrappers can be used to compose a complex pipeline with more than 1 fairness algorithm.
These wrappers can be used only with binary classifiers.
These fairness algorithms have been divided into 3 categories based on the parts in the pipeline that the algorithm can control. These 3 categories are Preprocessing, Postprocessing and Inprocessing[WIP].

## Preprocessing Algorithms
These are the algorithms that have control over the training data to be fed into machine learning model.
This class of algorithms improves the representation of groups in the training data.

### ReweighingSampling Algorithm
```@docs
ReweighingSamplingWrapper
MLJFair.ReweighingSamplingWrapper()
```

### Reweighing Algorithm
This model being wrapped with this wrapper needs to support weights. If the model doesn't support training using weights, then error is thrown. In case weights are not supported by your desired model, them switch to ReweighingSampling Algorithm.
To find the models in MLJ that support weights, execute:
```julia
using MLJ
models(x-> x.supports_weights)
```
```@docs
ReweighingWrapper
MLJFair.ReweighingWrapper()
MLJFair._calculateWeights
```

## Postprocessing
These are the algorithms that have control over the final predictions. They can tweak final predictions to optimise fairness constraints.

### Equalized Odds Algorithm
```@docs
EqOddsWrapper
MLJFair.EqOddsWrapper()
```

### LinProg Algorithm
This algorithm supports all the metrics provided by MLJFair.
```@docs
LinProgWrapper
MLJFair.LinProgWrapper()
```

## Composability

MLJFair provides you the ability to easily use multiple fairness algorithms on top of each other.
A fairness algorithm can be added over another fairness algorithm by simply wrapping the previous wrapped model with the new wrapper. MLJFair handles everything else for you!
The use of wrappers provides you the ability to add as many algorithms as you want!!

```@repl
using MLJFair, MLJ
X, y, _ = @load_toydata;
model = ConstantClassifier();
wrappedModel = ReweighingSamplingWrapper(classifier=model, grp=:Sex);
wrappedModel2 = EqOddsWrapper(classifier=wrappedModel, grp=:Sex);
mch = machine(wrappedModel2, X, y);
fit!(mch)
ŷ = predict(mch, X);
```
