using Terv

# interface flux between current conductors
# 1e7 should be the harmonic mean of hftrans/conductivity
function ccinterfaceflux!(src, phi_1,phi_2)
    for i in eachindex(phi_1)
        src[i] = (phi_2[i] - phi_1[i])
        src[i] *= 1e7
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
    ccinterfaceflux!(ct.crossterm_source,phi_s,value.(phi_t))
    ccinterfaceflux!(ct.crossterm_target,value.(phi_s),phi_t)
    #@. ct.crossterm_source = flux(phi_s,value.(phi_t))
    #@. ct.crossterm_target = flux(value.(phi_s), phi_t)
    #error("Cross term must be specialized for your equation and models. Did not understand how to specialize $target ($(typeof(target_model))) to $source ($(typeof(source_model)))")
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

    ccinterfaceflux!(ct.crossterm_source,phi_s,value.(phi_t))
    ccinterfaceflux!(ct.crossterm_target,value.(phi_s),phi_t)
    #@. ct.crossterm_source = flux(phi_s,value.(phi_t))
    #@. ct.crossterm_target = flux(value.(phi_s), phi_t)
    #error("Cross term must be specialized for your equation and models. Did not understand how to specialize $target ($(typeof(target_model))) to $source ($(typeof(source_model)))")
end

function regularizedSqrt(x, th)
    ind = (x <= th)
    if(!ind)
        y = x.^0.5
    else
        y = x/th*sqrt(th)
 
    end
    return y   
end

function reaction_rate(
    phi_a, c_a, R, ocd,
    phi_e, c_e, activematerial, electrolyte
    )
    T = 298.15 # for now
    n = nChargeCarriers(activematerial)
    cmax = cMax(activematerial) # how to get model
    vsa = volumetricSurfaceArea(activematerial)
    # ocd could have beencalculated of only this cells 
    eta = (phi_e - phi_a - ocd);
    th = 1e-3*cmax;
    j0 = R*regularizedSqrt(c_e*(cmax - c_a)*c_a, th)*n*FARADAY_CONST;
    R = vsa*ButlerVolmerEquation(j0, 0.5, n, eta, T);
    return R./(n*F);
end

function sourceElectricMaterial!(eS,eM,
    phi_a, c_a, R0,  ocd,
    phi_e, c_e, activematerial, electrolyte
    )
    #eS = similar(phi_a)
    #eM = similar(phi_a)
    for i in enumerate(phi_a)
        R = reaction_rate(phi_a[i], c_a[i], R0[i], ocd[i],
        phi_e[i], c_e[i], activematerial, electrolyte)
        vols =1.0 # volums of cells

        eS[i] = vols*R*n*F
        eM[i] = vols*R
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
    c_e = source_storage.state.C[ct.impact.target]
    c_a = target_storage.state.C[ct.impact.source]

    eM  = similar(ct.crossterm_target)
    sourceElectricMaterial!(ct.crossterm_target,eM,
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    #ct.crossterm_target = eE
    eM = similar(ct.crossterm_source)
    #eE, eM = 
    sourceElectricMaterial!(ct.crossterm_source,eM,
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
    
    #ct.crossterm_source = eE
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
    phi_e = source_storage.state.Phi[ct.impact.target]
    phi_a = target_storage.state.Phi[ct.impact.source]  
    ocd = target_storage.state.Ocd[ct.impact.source]
    R = target_storage.state.ReactionRateConst[ct.impact.source]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]
    eM = similar(ct.crossterm_source)
    #eE, eM = 
    sourceElectricMaterial!(ct.crossterm_source, eM,
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    #ct.crossterm_source = eE

    #eE, eM =
    eM = similar(ct.crossterm_target)
    sourceElectricMaterial!(ct.crossterm_target, eM,
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
       
    #ct.crossterm_target = eE

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
    phi_e = source_storage.state.Phi[ct.impact.target]
    phi_a = target_storage.state.Phi[ct.impact.source]  
    ocd = target_storage.state.Ocd[ct.impact.source]
    R = target_storage.state.ReactionRateConst[ct.impact.source]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]
    eE = similar(ct.crossterm_target)
    #eE, eM = 
    sourceElectricMaterial!(eE,ct.crossterm_target,
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    #et.crossterm_target = eM

    eE = similar(ct.crossterm_source)
    #eE, eM = 
    sourceElectricMaterial!(eE,ct.crossterm_source, 
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
    
    #ct.crossterm_source = eM
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
    eE = similar(ct.crossterm_source)
    #eE, eM = 
    sourceElectricMaterial!(eE,ct.crossterm_source, 
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    ct.crossterm_source = eM
    eE = similar(ct.crossterm_target)
    #eE, eM = 
    sourceElectricMaterial!(eE,ct.crossterm_target,
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )   
    #ct.crossterm_target = eM
end
