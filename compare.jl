using Printf
using FileIO
using LinearAlgebra

is_approx_or_same(a, b) = isapprox(a, b) || all(a .=== b)

for step in 1
    for substep in 1:10
        @info step substep

        name = @sprintf("s_%02d_%02d.jld2", step, substep)
        b = load("bad/$name")
        g = load("good/$name")

        # @assert all(b["Q"] .=== g["Q"])
        # @assert all(b["Qhat"] .=== g["Qhat"])
        # @assert all(b["Qtt"] .=== g["Qtt"])
        # for (bQs, gQs) in zip(b["Qstages"], g["Qstages"])
        #     @assert all(bQs .=== gQs)
        # end
        # for (bRs, gRs) in zip(b["Rstages"], g["Rstages"])
        #     @assert all(bRs .=== gRs)
        # end

        @assert is_approx_or_same(b["Q"], g["Q"])
        @assert is_approx_or_same(b["Qhat"], g["Qhat"])
        @assert is_approx_or_same(b["Qtt"], g["Qtt"])
        for (bQs, gQs) in zip(b["Qstages"], g["Qstages"])
            @assert is_approx_or_same(bQs, gQs)
        end
        for (bRs, gRs) in zip(b["Rstages"], g["Rstages"])
            @assert is_approx_or_same(bRs, gRs)
        end

    end
end
