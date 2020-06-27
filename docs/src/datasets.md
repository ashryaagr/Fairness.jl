# Fairness Datasets

To make it easy to try algorithms and metrics on various datasets, MLJFair shall be providing with the popular fairness datasets.

These datasets can be easily accesses using macros.

Following datasets shall be available in future through macros.
- COMPAS Dataset
- Adult Dataset
- German Dataset
- Bank Dataset

But currently, only a toy dataset is available.

## Toy Data
This is a 10 row dataset that was used by authors of Reweighing Algorithm.

```@docs
@load_toydata
@load_toyfairtensor
```

```@repl
using MLJFair
X, y, ŷ = @load_toydata;
X
y
ŷ
ft = @load_toyfairtensor
```

Below is the helper function that can be used to load custom dataset as well
```@docs
MLJFair.load_dataset
```
