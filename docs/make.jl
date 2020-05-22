Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using ClimateMachine.Microphysics, Documenter, Literate

generated_dir = joinpath(@__DIR__, "src", "generated") # generated files directory
mkpath(generated_dir)

pages = Any[
    "How-to-guides" => "Microphysics.md",
]

mathengine = MathJax(Dict(
    :TeX => Dict(
        :equationNumbers => Dict(:autoNumber => "AMS"),
        :Macros => Dict(),
    ),
))

format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    mathengine = mathengine,
    collapselevel = 1,
)

makedocs(
    sitename = "ClimateMachine",
    source="src/HowToGuides/Atmos",
    build="build/generated/HowToGuides/Atmos",
    doctest = false,
    strict = false,
    linkcheck = false,
    format = format,
    checkdocs = :none,
    clean = false,
    #modules = [ClimateMachine.Microphysics],
    #modules = [ClimateMachine],
    pages = pages,
)

include("clean_build_folder.jl")
