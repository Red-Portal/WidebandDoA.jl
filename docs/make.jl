
using Pkg

using WidebandDoA
using Documenter

DocMeta.setdocmeta!(WidebandDoA, :DocTestSetup, :(using WidebandDoA); recursive=true)


makedocs(;
    modules=[WidebandDoA],
    repo="https://github.com/Red-Portal/WidebandDoA.jl/blob/{commit}{path}#{line}",
    sitename="WidebandDoA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/WidebandDoA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"                   => "index.md",
        "General Usage"          => "general.md",
        "Inference"              => "inference.md",
        "Demonstration"          => "demonstration.md",
        "Validation of Baseline" => "baseline.md",
        "API"                    => "api.md"
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/WidebandDoA.jl",
    devbranch="main",
)
