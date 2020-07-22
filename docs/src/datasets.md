# Fairness Datasets

To make it easy to try algorithms and metrics on various datasets, Fairness.jl provides you with the popular fairness datasets.

These datasets can be easily accesses using macros.

### COMPAS Dataset
```@docs
@load_compas
```
```@repl datasets
using Fairness
X, y = @load_compas;
```

### Adult Dataset
```@docs
@load_adult
```

### German Credit Dataset
```@docs
@load_german
```

### Inspecting Datasets
To see the columns in dataset, their types and scientific types, you can use `schema` from MLJ.
```@repl
using Fairness, MLJ
X, y = @load_adult;
schema(X)
```

## Toy Data
This is a 10 row dataset that was used by authors of Reweighing Algorithm.
This dataset is intended to test ideas and evaluate metrics without calculating predictions.
It is different from other macros as it returns (X, y, ŷ) instead of (X, y)

```@docs
@load_toydata
@load_toyfairtensor
```

```@repl datasets
X, y, ŷ = @load_toydata;
ft = @load_toyfairtensor
```

## Other Datasets
You can try working with the vast range of datasets available through OpenML.
Refer [MLJ's OpenML documentation](https://alan-turing-institute.github.io/MLJ.jl/v0.9/openml_integration/) for the OpenML API.
The id to be passed to OpenML.load can be found through [OpenML site](https://www.openml.org/search?type=data)
```@repl
using MLJBase, Fairness
using DataFrames
data = OpenML.load(1480); # load Indian Liver Patient Dataset
df = DataFrame(data) ;
y, X = unpack(df, ==(:Class), name->true); # Unpack the data into features and target
y = coerce(y, Multiclass); # Specifies that the target y is of type Multiclass. It is othewise a string.
coerce!(X, :V2 => Multiclass, Count => Continuous); # Specifying which columns are Multiclass in nature. Converting from Count to Continuous enables use of more models.
```

## Helper Functions
```@docs
Fairness.ensure_download
```
