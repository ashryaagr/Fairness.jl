using Documenter
using MLJFair

makedocs(;
    modules=[MLJFair],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "Measures" => "measures.md",
        "FairTensor" => "fairtensor.md",
    ],
    repo="https://github.com/ashryaagr/MLJFair.jl/blob/{commit}{path}#L{line}",
    sitename="MLJFair.jl"
)

# By default Documenter does not deploy docs just for PR
# this causes issues with how we're doing things and ends
# up choking the deployment of the docs, so  here we
# force the environment to ignore this so that Documenter
# does indeed deploy the docs
ENV["TRAVIS_PULL_REQUEST"] = "false"

deploydocs(;
    repo="github.com/ashryaagr/MLJFair.jl.git",
)
