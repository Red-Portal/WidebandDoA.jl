using WidebandDoA
using Documenter

DocMeta.setdocmeta!(WidebandDoA, :DocTestSetup, :(using WidebandDoA); recursive=true)

makedocs(;
    modules=[WidebandDoA],
    authors="Ray Kim <msca8h@naver.com> and contributors",
    repo="https://github.com/Red-Portal/WidebandDoA.jl/blob/{commit}{path}#{line}",
    sitename="WidebandDoA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/WidebandDoA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"          => "index.md",
        "Demonstration" => "demonstration.md",
        "Baseline"      => "baseline.md",
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/WidebandDoA.jl",
    devbranch="main",
)
