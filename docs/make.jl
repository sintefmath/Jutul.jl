using Jutul
using Documenter

DocMeta.setdocmeta!(Jutul, :DocTestSetup, :(using Jutul); recursive=true)

makedocs(;
    modules=[Jutul],
    authors="Olav Møyner <olav.moyner@sintef.no> and contributors",
    repo="https://github.com/sintefmath/Jutul.jl/blob/{commit}{path}#{line}",
    sitename="Jutul.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sintefmath.github.io/Jutul.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sintefmath/Jutul.jl",
    devbranch="main",
)
