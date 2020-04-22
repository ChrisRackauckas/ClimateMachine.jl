# # Dry Atmosphere GCM diagnostics
# 
# This file computes selected diagnostics for the GCM and outputs them on the
# spherical interpolated diagnostic grid.
#
# Use it by calling `Diagnostics.setup_atmos_default_diagnostics()`.
#
# TODO:
# - the struct functions need to be used to generalise the choices
#   - these require Array(FT, 1) but interpolation requires FT
#   - maybe will need to define a conversion function? struct(num_thermo(FT), Nel)(vari) --> Array(num_thermo(FT), vari, Nel)
# - elementwise aggregation of interpolated vars very slow
# - enable zonal means and calculation of covariances using those means
# - add more variables, including hioriz streamfunction from laplacial of vorticity (LN)
# - density weighting
# - maybe change thermo/dun separation to local/nonlocal vars?

using Printf
using Statistics

using ..Atmos
using ..Atmos: thermo_state, turbulence_tensors
using ..DGmethods
using ..DGmethods: vars_state_conservative, vars_state_auxiliary
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics

using CLIMAParameters.Atmos.SubgridScale: inv_Pr_turb
using LinearAlgebra

function atmos_gcm_default_init(dgngrp::DiagnosticsGroup, currtime)
    if !(dgngrp.interpol isa InterpolationCubedSphere)
        @warn """
            Diagnostics ($dgngrp.name): currently requires `InterpolationCubedSphere`!
            """
    end
end

# 3D variables
function vars_atmos_gcm_default_simple_3d(atmos::AtmosModel, FT)
    @vars begin
        u::FT
        v::FT
        w::FT
        rho::FT
        temp::FT
        thd::FT                 # θ_dry
        thv::FT                 # θ_vir
        et::FT                  # e_tot
        ei::FT                  # e_int
        ht::FT
        hm::FT

        vort::FT                # Ω₃

        moisture::vars_atmos_gcm_default_simple_3d(atmos.moisture, FT)
    end
end
vars_atmos_gcm_default_simple_3d(::MoistureModel, FT) = @vars()
function vars_atmos_gcm_default_simple_3d(m::EquilMoist, FT)
    @vars begin
        qt::FT                  # q_tot
        ql::FT                  # q_liq
        qv::FT                  # q_vap
        qi::FT                  # q_ice
        thl::FT                 # θ_liq
    end
end
num_atmos_gcm_default_simple_3d_vars(m, FT) =
    varsize(vars_atmos_gcm_default_simple_3d(m, FT))
atmos_gcm_default_simple_3d_vars(m, array) =
    Vars{vars_atmos_gcm_default_simple_3d(m, eltype(array))}(array)

function atmos_gcm_default_simple_3d_vars!(
    atmos::AtmosModel,
    state_conservative,
    thermo,
    dyni,
    vars,
)
    vars.u = state_conservative.ρu[1] / state_conservative.ρ
    vars.v = state_conservative.ρu[2] / state_conservative.ρ
    vars.w = state_conservative.ρu[3] / state_conservative.ρ
    vars.rho = state_conservative.ρ
    vars.temp = thermo.T
    vars.thd = thermo.θ_dry
    vars.thv = thermo.θ_vir
    vars.et = state_conservative.ρe / state_conservative.ρ
    vars.ei = thermo.e_int
    vars.ht = thermo.h_tot
    vars.hm = thermo.h_moi

    vars.vort = dyni.Ω₃

    atmos_gcm_default_simple_3d_vars!(
        atmos.moisture,
        state_conservative,
        thermo,
        vars,
    )

    return nothing
end
function atmos_gcm_default_simple_3d_vars!(
    ::MoistureModel,
    state_conservative,
    thermo,
    vars,
)
    return nothing
end
function atmos_gcm_default_simple_3d_vars!(
    moist::EquilMoist,
    state_conservative,
    thermo,
    vars,
)
    vars.moisture.qt = state_conservative.moisture.ρq_tot / state_conservative.ρ
    vars.moisture.ql = thermo.moisture.q_liq
    vars.moisture.qv = thermo.moisture.q_vap
    vars.moisture.qi = thermo.moisture.q_ice
    vars.moisture.thl = thermo.moisture.θ_liq_ice

    return nothing
end

# Dynamic variables
function vars_dyn(FT)
    @vars begin
        Ω₁::FT
        Ω₂::FT
        Ω₃::FT
    end
end
dyn_vars(array) = Vars{vars_dyn(eltype(array))}(array)

# zonal means
#ds.T_zm = mean(.*1., ds.T; dims = 3)
#ds.u_zm = mean((ds.u); dims = 3 )
#v_zm = mean(ds.v; dims = 3)
#w_zm = mean(ds.w; dims = 3)

# (co)variances
#ds.uvcovariance = (ds.u .- ds.u_zm) * (ds.v .- v_zm)
#ds.vTcovariance = (ds.v .- v_zm) * (ds.T .- ds.T_zm)

"""
    atmos_gcm_default_collect(bl, currtime)

    Master function that performs a global grid traversal to compute various
    diagnostics using the above functions.
"""
function atmos_gcm_default_collect(dgngrp::DiagnosticsGroup, currtime)
    interpol = dgngrp.interpol
    if !(interpol isa InterpolationCubedSphere)
        @warn """
            Diagnostics ($dgngrp.name): currently requires `InterpolationCubedSphere`!
            """
        return nothing
    end

    DA = ClimateMachine.array_type()
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    bl = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get needed arrays onto the CPU
    if Array ∈ typeof(Q).parameters
        host_state_conservative = Q.realdata
        host_state_auxiliary = dg.state_auxiliary.realdata
    else
        host_state_conservative = Array(Q.realdata)
        host_state_auxiliary = Array(dg.state_auxiliary.realdata)
    end
    FT = eltype(host_state_conservative)

    # TODO: can this be done in one pass?
    #
    # Non-local vars, e.g. relative vorticity
    vgrad = compute_vec_grad(bl, Q, dg)
    vort = compute_vorticity(dg, vgrad)

    # TODO: make this like the above
    # Compute thermo variables
    thermo_array = Array{FT}(undef, npoints, num_thermo(bl, FT), nrealelem)
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state_conservative =
            extract_state_conservative(dg, host_state_conservative, ijk, e)
        state_auxiliary =
            extract_state_auxiliary(dg, host_state_auxiliary, ijk, e)

        thermo = thermo_vars(bl, view(thermo_array, ijk, :, e))
        compute_thermo!(bl, state_conservative, state_auxiliary, thermo)
    end

    # Interpolate the state, thermo and dyn vars to sphere (u and vorticity
    # need projection to zonal, merid). All this may happen on the GPU.
    istate =
        DA(Array{FT}(undef, interpol.Npl, number_state_conservative(bl, FT)))
    interpolate_local!(interpol, Q.realdata, istate)

    ithermo = DA(Array{FT}(undef, interpol.Npl, num_thermo(bl, FT)))
    interpolate_local!(interpol, DA(thermo_array), ithermo)

    idyn = DA(Array{FT}(undef, interpol.Npl, size(vort.data, 2)))
    interpolate_local!(interpol, vort.data, idyn)

    # TODO: get indices here without hard-coding them
    _ρu, _ρv, _ρw = 2, 3, 4
    project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
    _Ω₁, _Ω₂, _Ω₃ = 1, 2, 3
    project_cubed_sphere!(interpol, idyn, (_Ω₁, _Ω₂, _Ω₃))

    # FIXME: accumulating to rank 0 is not scalable
    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
    all_thermo_data = accumulate_interpolated_data(mpicomm, interpol, ithermo)
    all_dyn_data = accumulate_interpolated_data(mpicomm, interpol, idyn)

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)
        dim_names = tuple(collect(keys(dims))...)

        # combine state, thermo and dyn variables, and their manioulations on the interpolated grid
        nlon = length(dims["long"])
        nlat = length(dims["lat"])
        nrad = length(dims["rad"])

        simple_3d_vars_array = Array{FT}(
            undef,
            nlon,
            nlat,
            nrad,
            num_atmos_gcm_default_simple_3d_vars(bl, FT),
        )

        @visitIQ nlon nlat nrad begin
            statei = Vars{vars_state_conservative(bl, FT)}(view(
                all_state_data,
                lo,
                la,
                le,
                :,
            ))
            thermoi = thermo_vars(bl, view(all_thermo_data, lo, la, le, :))
            dyni = dyn_vars(view(all_dyn_data, lo, la, le, :))
            simple_3d_vars = atmos_gcm_default_simple_3d_vars(
                bl,
                view(simple_3d_vars_array, lo, la, le, :),
            )

            atmos_gcm_default_simple_3d_vars!(
                bl,
                statei,
                thermoi,
                dyni,
                simple_3d_vars,
            )
        end

        # attribute names to the vars in dsumsi and collect in a dict
        varvals = OrderedDict()
        varnames = flattenednames(vars_atmos_gcm_default_simple_3d(bl, FT))
        for vari in 1:length(varnames)
            varvals[varnames[vari]] =
                (dim_names, simple_3d_vars_array[:, :, :, vari])
        end

        # write output
        dprefix = @sprintf(
            "%s_%s_%s_num%04d",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
            dgngrp.num
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
        write_data(dgngrp.writer, dfilename, dims, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_gcm_default_fini(dgngrp::DiagnosticsGroup, currtime) end
