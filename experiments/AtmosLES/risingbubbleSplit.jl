using ClimateMachine
ClimateMachine.init()

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.MoistThermodynamics
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK
using ClimateMachine.Atmos: vars_state_conservative, vars_state_auxiliary

using StaticArrays
using Test
using Printf
using MPI

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# ------------------------ Description ------------------------- #
# 1) Dry Rising Bubble (circular potential temperature perturbation)
# 2) Boundaries - `All Walls` : Impenetrable(FreeSlip())
#                               Laterally periodic
# 3) Domain - 2500m[horizontal] x 2500m[horizontal] x 2500m[vertical]
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 1:1
# 7) Overrides defaults for
#               `init_on_cpu`
#               `solver_type`
#               `sources`
#               `C_smag`
# 8) Default settings can be found in `src/Driver/Configurations.jl`
# ------------------------ Description ------------------------- #
function init_risingbubble!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    γ::FT = c_p / c_v
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)

    xc::FT = 1250
    yc::FT = 1250
    zc::FT = 1000
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    rc::FT = 500
    θ_ref::FT = 300
    Δθ::FT = 0

    if r <= rc
        Δθ = FT(5) * cospi(r / rc / 2)
    end

    #Perturbed state:
    θ = θ_ref + Δθ # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas) # density
    q_tot = FT(0)
    ts = LiquidIcePotTempSHumEquil(bl.param_set, θ, ρ, q_tot)
    q_pt = PhasePartition(ts)

    ρu = SVector(FT(0), FT(0), FT(0))

    #State (prognostic) variable assignment
    e_kin = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_pt.tot
end

function config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

    # Choose explicit solver

    ode_solver = ClimateMachine.MISSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = MIS2,
        fast_method = (dg,Q) -> StormerVerlet(dg, [1,5], 2:4, Q),
        number_of_steps = (10,),
    )

    # Set up the model
    C_smag = FT(0.23)
    ref_state =
        HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0))
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = SmagorinskyLilly{FT}(C_smag),
        source = (Gravity(),),
        ref_state = ref_state,
        init_state_conservative = init_risingbubble!,
    )

    # Problem configuration
    config = ClimateMachine.AtmosLESConfiguration(
        "DryRisingBubble",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_risingbubble!,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    ClimateMachine.init()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 2
    # Domain resolution and size
    Δh = FT(50)
    Δv = FT(50)
    resolution = (Δh, Δh, Δv)
    # Domain extents
    xmax = FT(2500)
    ymax = FT(2500)
    zmax = FT(2500)
    # Simulation time
    t0 = FT(0)
    timeend = FT(1000)

    # Courant number
    CFL = FT(20)

    driver_config = config_risingbubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = 0.05,
        Courant_number = CFL,
    )
    dgn_config = config_diagnostics(driver_config)

    # User defined filter (TMAR positivity preserving filter)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    vtk_step = 0
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1)  do (init=false)
        mkpath("./vtk-rtb/")
        outprefix = @sprintf("./vtk-rtb/small_mpirank%04d_step%04d",
                         MPI.Comm_rank(driver_config.mpicomm), vtk_step)
        writevtk(outprefix, solver_config.Q, solver_config.dg,
            flattenednames(vars_state_conservative(driver_config.bl,FT)),
            solver_config.dg.state_auxiliary,
            flattenednames(vars_state_auxiliary(driver_config.bl,FT)))
        vtk_step += 1
        nothing
     end

    # Invoke solver (calls solve! function for time-integrator)
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,cbvtk),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
