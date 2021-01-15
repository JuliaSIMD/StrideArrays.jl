using StrideArrays
using Documenter

makedocs(;
    modules=[StrideArrays],
    authors="Chris Elrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/chriselrod/StrideArrays.jl/blob/{commit}{path}#L{line}",
    sitename="StrideArrays.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/StrideArrays.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chriselrod/StrideArrays.jl",
)
