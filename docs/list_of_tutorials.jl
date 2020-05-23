####
#### Defines list of tutorials given `generated_dir`
####

generate_tutorials = true

tutorials = Any[]

# Allow flag to skip generated
# tutorials since this is by
# far the slowest part of the
# docs build.
if generate_tutorials

    # generate tutorials
    import Literate

    include("gather_pages.jl")

    tutorials_dir = joinpath(@__DIR__, "..", "tutorials")

    tutorials_jl = [
        joinpath("Atmos","dry_rayleigh_benard.jl"),
        joinpath("Atmos","heldsuarez.jl"),
        joinpath("Atmos","risingbubble.jl"),
        joinpath("Land","Heat","heat_equation.jl"),
        joinpath("Microphysics","ex_1_saturation_adjustment.jl"),
        joinpath("Microphysics","ex_2_Kessler.jl"),
        joinpath("Numerics","DGmethods","nonnegative.jl"),
        joinpath("Numerics","LinearSolvers","bgmres.jl"),
        joinpath("Numerics","LinearSolvers","cg.jl"),
        joinpath("literate_markdown.jl"),
        joinpath("topo.jl"),
    ]


    tutorials_jl = [joinpath(tutorials_dir, x) for x in tutorials_jl]

    skip_execute = [
        joinpath("Atmos","dry_rayleigh_benard.jl"),                 # takes too long
        joinpath("Atmos","heldsuarez.jl"),                          # broken
        joinpath("Atmos","risingbubble.jl"),                        # broken
        joinpath("Microphysics","ex_1_saturation_adjustment.jl"),   # too long
        joinpath("Microphysics","ex_2_Kessler.jl"),                 # too long
        joinpath("topo.jl"),                                        # broken
    ]

    println("Building literate tutorials:")
    for tutorial in tutorials_jl
        println("    $(tutorial)")
    end

    for tutorial in tutorials_jl
        gen_dir =
            joinpath(generated_dir, relpath(dirname(tutorial), tutorials_dir))
        input = abspath(tutorial)
        script = Literate.script(input, gen_dir)
        code = strip(read(script, String))
        mdpost(str) = replace(str, "@__CODE__" => code)
        Literate.markdown(input, gen_dir, postprocess = mdpost)
        if !any(occursin.(skip_execute, Ref(input)))
            Literate.notebook(input, gen_dir, execute = true)
        end
    end

    tutorials, filenames = gather_pages(;
        filenames = relpath.(tutorials_jl, dirname(@__DIR__)),
        transform_extension=x->replace_reverse(x, ".jl" => ".md"; count=1),
        transform_path=x -> replace(x, "tutorials" => "generated", count = 1),
    )

end
