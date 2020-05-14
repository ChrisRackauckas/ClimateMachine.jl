using Printf
using FileIO
using LinearAlgebra

# check(a, b) = all(isapprox.(a, b)) || all(a .=== b)
check(a, b) = norm((a - b) ./ maximum(a, dims = (1, 3)), Inf)

for step in 1
    for substep in 7
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

        @show check(b["Q"], g["Q"])

        @show check(b["Qhat"], g["Qhat"])

        @show check(b["Qtt"], g["Qtt"])

        @show check(b["Q"][:, 1:(end - 1), :], g["Q"][:, 1:(end - 1), :])

        @show check(b["Qtt"][:, 1:(end - 1), :], g["Qtt"][:, 1:(end - 1), :])

        @show check(b["Qtt"][:, end, :], b["Qhat"][:, end, :])

        # Why is this so different?
        @show check(g["Qtt"][:, end, :], g["Qhat"][:, end, :])

        @show check(b["Qtt"][:, end, :], g["Qtt"][:, end, :])

        # @show check(b["Qtt"][:,1,:], g["Qtt"][:,1,:])
        # @show check(b["Qtt"][:,2,:], g["Qtt"][:,2,:])
        # @show check(b["Qtt"][:,3,:], g["Qtt"][:,3,:])
        # @show check(b["Qtt"][:,4,:], g["Qtt"][:,4,:])
        # @show check(b["Qtt"][:,5,:], g["Qtt"][:,5,:])
        # @show check(b["Qtt"][:,end,:], g["Qtt"][:,end,:])

        for (bQs, gQs) in zip(b["Qstages"], g["Qstages"])
            @show check(bQs, gQs)
        end
        for (bRs, gRs) in zip(b["Rstages"], g["Rstages"])
            @show check(bRs, gRs)
        end

    end
end
