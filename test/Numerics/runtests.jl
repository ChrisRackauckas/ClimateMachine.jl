using Test, Pkg

@testset "Numerics" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["Mesh", "DGMethods", "LinearSolvers", "ODESolvers"]
        if all_tests || "$submodule" in ARGS || "Numerics" in ARGS
            include_test(submodule)
        end
    end

end
