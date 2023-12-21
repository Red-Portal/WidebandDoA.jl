using WideBandDOA
using Documenter

DocMeta.setdocmeta!(WideBandDOA, :DocTestSetup, :(using WideBandDOA); recursive=true)

makedocs(;
    modules=[WideBandDOA],
    authors="Ray Kim <msca8h@naver.com> and contributors",
    repo="https://github.com/Red-Portal/WideBandDOA.jl/blob/{commit}{path}#{line}",
    sitename="WideBandDOA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/WideBandDOA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/WideBandDOA.jl",
    devbranch="main",
)
