using Documenter
using Fairness
using MLJBase, MLJ
using PrettyPrinting
using DataFrames

makedocs(;
    modules=[Fairness],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "FairTensor" => "fairtensor.md",
        "Measures" => "measures.md",
        "Fairness Algortihms" => "algorithms.md",
        "Datasets" => "datasets.md"
    ],
    repo="https://github.com/ashryaagr/Fairness.jl/blob/{commit}{path}#L{line}",
    sitename="Fairness"
)

# By default Documenter does not deploy docs just for PR
# this causes issues with how we're doing things and ends
# up choking the deployment of the docs, so  here we
# force the environment to ignore this so that Documenter
# does indeed deploy the docs
ENV["TRAVIS_PULL_REQUEST"] = "false"

deploydocs(;
    repo="github.com/ashryaagr/Fairness.jl.git",
)
