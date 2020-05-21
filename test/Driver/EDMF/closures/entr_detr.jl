function source!(
    ss::SingleStack{FT, N},
    m::EntrainmentDetrainment,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) where {FT, N}

    # Alias convention:
    gm = state
    en = state
    up = state.edmf.updraft
    gm_s = source
    en_s = source
    up_s = source.edmf.updraft

    ε = 1
    # for i in 1:N
    #     u_i = up[i].ρu/gm.ρ
    #     up_s[i].ρa += up[i].ρa * u_i * ε
    # end
    return ε

end;
