#!/usr/bin/env julia
#
# Called with no arguments will create the system image
#     ClimateMachine.so
# in the `@__DIR__` directory.
#
# Called with a single argument the system image will be placed in the path
# specified by the argument (relative to the callers path)
#
# Called with a specified systemimg path and `true`, the system image will
# compile the climate machine package module (useful for CI)

sysimage_path =
    len(ARGS) < 1 ? joinpath(@__DIR__, "ClimateMachine.so") : abspath(ARGS[1])

climatemachine_pkg = len(ARGS) > 1 && ARGS[2] == "true" ? true : false

using Pkg
Pkg.add("PackageCompiler")

using PackageCompiler
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

pkgs::Vector{Symbol} = []
if climatemachine_pkg
    append!(pkgs, :ClimateMachine)
else
    if VERSION < v"1.4"
        append!(pkgs, keys(Pkg.installed()))
    else
        append!(pkgs, [v.name for v in values(Pkg.dependencies())])
    end
end
delete!(pkgs, "Pkg")

PackageCompiler.create_sysimage(
    pkgs,
    sysimage_path = sysimage_path,
    precompile_execution_file = joinpath(
        @__DIR__,
        "..",
        "..",
        "test",
        "Numerics",
        "DGmethods",
        "Euler",
        "isentropicvortex.jl",
    ),
)

@info "Created system image object file: '$(sysimage_path)'"
