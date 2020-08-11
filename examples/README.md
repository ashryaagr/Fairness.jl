# Examples

This directory contains examples in the form of jupyter notebooks and julia scripts.

## Generating Notebooks and Markdown files from julia scripts

Run the file `examples/literate.jl` to get both, the Markdown file and the jupyter notebook for the examples that are in the form of Julia scripts.

```julia
julia --color=yes --project=docs/ -e'
	using Pkg
	Pkg.develop(PackageSpec(path=pwd()))
	Pkg.instantiate()
	include("docs/make.jl")'
```

## List of Examples
- [nextjournal.jl](nextjournal.jl) : Introductory End-to-end example of using Fairness.jl with NeuralNetowork Classifier. The notebook is available at https://nextjournal.com/ashryaagr/fairness
- [algorithms.ipynb](algorithms.ipynb) : Tutorial for using Fairness Algorithms
- [metrics.ipynb](metrics.ipynb) : Tutorial for using Fairness Metrics.

## Adding new example as a Julia Script

Add the example in `examples/` and add following lines in literate.jl
```julia
Literate.markdown("your_julia_script.jl", "."; documenter=false)
Literate.notebook("your_julia_script.jl", ".")
```
Also update the list of examples in this README.
