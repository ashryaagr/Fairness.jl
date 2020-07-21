# Fairness

[Fairness](https://github.com/ashryaagr/Fairness.jl) is a bias audit and mitigation toolkit in julia and is supported by MLJ Ecosystem.

# Installation
```julia
julia> using Pkg
julia> Pkg.activate("my_environment", shared=true)
julia> Pkg.add("https://github.com/ashryaagr/Fairness.jl")
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
  - InProcessing Algorithms [WIP]
  - PostProcessing Algorithms
- Fairness Datasets (Macros)
