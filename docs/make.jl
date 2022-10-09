using StrideArrays
using Documenter

makedocs(;
  modules = [StrideArrays],
  authors = "Chris Elrod",
  repo = "https://github.com/JuliaSIMD/StrideArrays.jl/blob/{commit}{path}#L{line}",
  sitename = "StrideArrays.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSIMD.github.io/StrideArrays.jl",
    assets = String[],
  ),
  pages = [
    "Home" => "index.md",
    "Getting Started" => "getting_started.md",
    "Architecture Benchmarks" =>
      ["arches/cascadelake.md", "arches/tigerlake.md", "arches/haswell.md"],
    "Random Number Generation" => "rng.md",
    "Broadcasting" => "broadcasting.md",
    "Stack Allocation" => "stack_allocation.md",
  ],
  strict = false,
)

deploydocs(; repo = "github.com/JuliaSIMD/StrideArrays.jl")
