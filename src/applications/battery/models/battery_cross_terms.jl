# interface flux between current conductors
# 1e7 should be the harmonic mean of hftrans/conductivity
function ccinterfaceflux!(src, phi_1,phi_2)
    for i in eachindex(phi_1)
        src[i] = (phi_2[i] - phi_1[i])
        src[i] *= -1e7
    end
end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Charge}, 
    target_storage,
    source_storage,
    target_model::SimulationModel{<:Any, TT, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, <:Any, <:Any, <:Any}, 
    target, source, dt
    ) where {TT <: CurrentCollector} # or TS <: CurrentCollector

    phi_t = target_storage.state.Phi[ct.impact.target]
    phi_s = source_storage.state.Phi[ct.impact.source]

    ccinterfaceflux!(ct.crossterm_source, phi_s, value.(phi_t))
    ccinterfaceflux!(ct.crossterm_target, value.(phi_s), phi_t)
end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Charge}, 
    target_storage,
    source_storage,
    target_model::SimulationModel{<:Any, <:Any, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    target, source, dt
    ) where {TS <: CurrentCollector}

    phi_t = target_storage.state.Phi[ct.impact.target]
    phi_s = source_storage.state.Phi[ct.impact.source]

    ccinterfaceflux!(ct.crossterm_source, phi_s, value.(phi_t))
    ccinterfaceflux!(ct.crossterm_target, value.(phi_s), phi_t)
end

function regularizedSqrt(x, th)
    ind = (x <= th)
    if !ind
        y = x^0.5
    else
        y = x/th*sqrt(th)
    end
    return y   
end

function butlerVolmerEquation(j0, alpha, n, eta, T)
    res = j0 * (
        exp(  alpha * n * FARADAY_CONST * eta / (GAS_CONSTANT * T ) ) - 
        exp( -(1-alpha) * n * FARADAY_CONST * eta / ( GAS_CONSTANT * T ) ) 
        )
    return res                   
end

function reaction_rate(
    phi_a, c_a, R0, ocd, T,
    phi_e, c_e, activematerial, electrolyte
    )

    n = nChargeCarriers(activematerial)
    cmax = cMax(activematerial) # how to get model
    vsa = volumetricSurfaceArea(activematerial)

    # ocd could have beencalculated of only this cells 
    eta = (phi_a - phi_e - ocd);
    th = 1e-3*cmax;
    j0 = R0*regularizedSqrt(c_e*(cmax - c_a)*c_a, th)*n*FARADAY_CONST;
    R = vsa*butlerVolmerEquation(j0, 0.5, n, eta, T);

    return R./(n*FARADAY_CONST);
end

function sourceElectricMaterial!(
    eS, eM, vols, T,
    phi_a, c_a, R0,  ocd,
    phi_e, c_e, activematerial, electrolyte
    )

    n = nChargeCarriers(activematerial)
    for (i, val) in enumerate(phi_a)
        # ! Hack, as we get error in ForwardDiff without .value
        # ! This will cause errors if T is not just constant
        temp = T[i].value
        R = reaction_rate(
            phi_a[i], c_a[i], R0[i], ocd[i], temp,
            phi_e[i], c_e[i], activematerial, electrolyte
            )
    
        eS[i] = -1.0 * vols[i] * R * n * FARADAY_CONST
        eM[i] = +1.0 * vols[i] * R
    end
    return (eS, eM)
end


function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Charge}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, SS, <:Any, <:Any}, 
    target, source, dt
    ) where {SS <: ActiveMaterial, TS <: Electrolyte} 

    activematerial = source_model.system
    electrolyte = target_model.system
    phi_e = target_storage.state.Phi[ct.impact.target]
    phi_a = source_storage.state.Phi[ct.impact.source]  
    ocd = source_storage.state.Ocd[ct.impact.source]
    R = source_storage.state.ReactionRateConst[ct.impact.source]
    c_e = target_storage.state.C[ct.impact.target]
    c_a = source_storage.state.C[ct.impact.source]
    volume = source_model.domain.grid.volumes
    T = source_storage.state.T[ct.impact.source]

    eM  = similar(ct.crossterm_source)
    sourceElectricMaterial!(
        ct.crossterm_source, eM, volume, T,
        phi_a, c_a, R, ocd,
        value.(phi_e), value.(c_e),
        activematerial, electrolyte  
    )

    eM = similar(ct.crossterm_target)
    sourceElectricMaterial!(
        ct.crossterm_target, eM, volume, T,
        value.(phi_a), value.(c_a), value.(R), value.(ocd),
        phi_e, c_e,
        activematerial, electrolyte  
    )
    ct.crossterm_target .*= -1.0
    ct.crossterm_source .*= -1.0
 end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Charge}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, SS, <:Any, <:Any}, 
    target, source, dt
    ) where {TS <: ActiveMaterial, SS <:Electrolyte}
    
    activematerial = target_model.system
    electrolyte = source_model.system 
    phi_e = source_storage.state.Phi[ct.impact.source]
    phi_a = target_storage.state.Phi[ct.impact.target]  
    ocd = target_storage.state.Ocd[ct.impact.target]
    R = target_storage.state.ReactionRateConst[ct.impact.target]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]
    volume = target_model.domain.grid.volumes
    T = target_storage.state.T[ct.impact.target]

    eM = similar(ct.crossterm_target)
    
    sourceElectricMaterial!(
        ct.crossterm_target, eM, volume, T,
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    eM = similar(ct.crossterm_source)
    sourceElectricMaterial!(
        ct.crossterm_source, eM, volume, T,
        value.(phi_a), value.(c_a), value.(R), value.(ocd),
        phi_e, c_e,
        activematerial, electrolyte  
    )
end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Mass}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, SS, <:Any, <:Any}, 
    target, source, dt
    ) where {TS <: ActiveMaterial, SS <:Electrolyte}

    activematerial = target_model.system
    electrolyte = source_model.system
    phi_e = source_storage.state.Phi[ct.impact.source]
    phi_a = target_storage.state.Phi[ct.impact.target]  
    ocd = target_storage.state.Ocd[ct.impact.target]
    R = target_storage.state.ReactionRateConst[ct.impact.target]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]
    volume = target_model.domain.grid.volumes
    T = target_storage.state.T[ct.impact.target]

    eE = similar(ct.crossterm_target)
    sourceElectricMaterial!(
        eE, ct.crossterm_target, volume, T,
        phi_a, c_a, R, ocd,
        value.(phi_e), value.(c_e),
        activematerial, electrolyte  
    )


    eE = similar(ct.crossterm_source)
    sourceElectricMaterial!(
        eE, ct.crossterm_source, volume, T,
        value.(phi_a), value.(c_a), value.(R), value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
 end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Mass}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, SS, <:Any, <:Any}, 
    target, source, dt
    ) where {SS <: ActiveMaterial, TS <:Electrolyte}

    activematerial = source_model.system
    electrolyte = target_model.system
    phi_e = target_storage.state.Phi[ct.impact.target]
    phi_a = source_storage.state.Phi[ct.impact.source]
    ocd = source_storage.state.Ocd[ct.impact.source]
    R = source_storage.state.ReactionRateConst[ct.impact.source]
    c_a = source_storage.state.C[ct.impact.source]
    c_e = target_storage.state.C[ct.impact.target]
    volume = source_model.domain.grid.volumes
    T = source_storage.state.T[ct.impact.source]

    eE = similar(ct.crossterm_source)
    sourceElectricMaterial!(
        eE, ct.crossterm_source, volume, T,
        phi_a, c_a, R, ocd,
        value.(phi_e), value.(c_e),
        activematerial, electrolyte  
    )

    eE = similar(ct.crossterm_target)
    sourceElectricMaterial!(
        eE, ct.crossterm_target, volume, T,
        value.(phi_a), value.(c_a), value.(R), value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
    ct.crossterm_target .*= -1.0
    ct.crossterm_source .*= -1.0  
    #ct.crossterm_target = eM
end
