# Fairness

[Fairness.jl](https://github.com/ashryaagr/Fairness.jl) is a bias audit and mitigation toolkit in julia and is supported by MLJ Ecosystem.

# Installation
```julia
julia> using Pkg
julia> Pkg.activate("my_environment", shared=true)
julia> Pkg.add("Fairness")
julia> Pkg.add("MLJ")
```

# Components
It shall be divided into following components
- FairTensor
- Measures
  - CalcMetrics
  - FairMetrics
  - BoolMetrics
- Algorithms
  - Preprocessing Algorithms
  - InProcessing Algorithms
  - PostProcessing Algorithms
- Fairness Datasets (Macros)

# Important Points to Note

After you go through the documentation or have a basic idea of the package, revisit the following points which are essential to make best out this package.

- Almost every fairness dataset has a categorical field. To be able to use the various MLJ Models, you should also use a ContinuousEncoder in the following manner.
```julia
model = @pipeline ContinuousEncoder @load(SomeClassifier, pkg=PackageOfClassifier)
```

- To get a list of classifiers that can be used with Fairness.jl, execute
```julia
using MLJ
models(x->x.target_scitype<:AbstractVector{<:Finite})
```
After you get the list, pick a classifier of your choice. Lets say you choose the tuple with `(name=RandomForestClassifier, package_name=DecisionTree)`.
Then to use the model, you have to execute
```julia
model = @pipeline ContinuousEncoder @load(RandomForestClassifier, pkg=DecisionTree)
```
Note that you might be asked to install a specific package for the classifier. Execute the instruction of the type `import Pkg; Pkg.add("--")`, which you will be asked to do when you try to load the model.
