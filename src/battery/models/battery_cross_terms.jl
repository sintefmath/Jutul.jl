using Terv


function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Charge}, 
    target_storage, source_storage, 
    target_model, source_model, 
    target, source, dt
    )
    phi_t = target_storage.state.Phi[ct.impact.target]
    phi_s = source_storage.state.Phi[ct.impact.source]
    function interfaceflux!(src, phi_1,phi_2)
        for i in eachindex(phi_1)
            src[i] = (phi_2[i] - phi_1[i])
            src[i] *= 1e7
        end
    end
    interfaceflux!(ct.crossterm_source,phi_s,value.(phi_t))
    interfaceflux!(ct.crossterm_target,value.(phi_s),phi_t)
    #@. ct.crossterm_source = flux(phi_s,value.(phi_t))
    #@. ct.crossterm_target = flux(value.(phi_s), phi_t)
    #error("Cross term must be specialized for your equation and models. Did not understand how to specialize $target ($(typeof(target_model))) to $source ($(typeof(source_model)))")
end

function regularizedSqrt(x, th)
    y = x # quick way to create y of same dimension as x and also preserved AD
    for i in range(x)
        ind = (x[i] <= th[i]);
        if(!ind)
            y[i] = x[i].^0.5
        else
            y[i] = x[i]/th*sqrt(th)
        end
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
    j0 = k.*regularizedSqrt(c_e.*(cmax - c).*c, th)*n*F;
    R = vsa.*ButlerVolmerEquation(j0, 0.5, n, eta, T);
    return R/(n*F);
end

function sourceElectricMaterial(
    phi_a, c_a, R,  ocd,
    phi_e, c_e, activematerial, electrolyte
    )
    R = reaction_rate(phi_a, c_a, R, ocd, phi_e, c_e, activematerial, electrolyte)
    vols =1.0 # volums of cells

    eS = vols.*R*n*F
    eM = vols.*R
    return (eS, eM)
end


function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Charge}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    target, source, dt
    ) where TS <: ActiveMaterial
    activematerial = TS
    electrolyte = SS 
    phi_e = target_storage.state.Phi[ct.impact.target]
    phi_a = source_storage.state.Phi[ct.impact.source]  
    ocd = source_storage.state.Ocd[ct.impact.source]
    R = source_storage.state.ReactionRateConst[ct.impact.source]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]

    eE, eM = sourceElectricMaterial(
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    ct.crossterm_target = eE

    eE, eM = sourceElectricMaterial(
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
    
    ct.crossterm_source = eE
 end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Charge}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    target, source, dt
    ) where {TS <: ActiveMaterial, SS <:ElectrolyteModel}
    
    activematerial = TS
    electrolyte = SS 
    phi_e = target_storage.state.Phi[ct.impact.target]
    phi_a = source_storage.state.Phi[ct.impact.source]  
    ocd = source_storage.state.Ocd[ct.impact.source]
    R = source_storage.state.ReactionRateConst[ct.impact.source]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]

    eE, eM = sourceElectricMaterial(
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    ct.crossterm_source = eE

    eE, eM = sourceElectricMaterial(
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
    
    ct.crossterm_target = eE

end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Mass}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    target, source, dt
    ) where {TS <: ActiveMaterial, SS <:ElectrolyteModel}

    activematerial = TS
    electrolyte = SS 
    phi_e = target_storage.state.Phi[ct.impact.target]
    phi_a = source_storage.state.Phi[ct.impact.source]  
    ocd = source_storage.state.Ocd[ct.impact.source]
    R = source_storage.state.ReactionRateConst[ct.impact.source]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]

    eE, eM = sourceElectricMaterial(
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    ct.crossterm_target = eM

    eE, eM = sourceElectricMaterial(
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
    
    ct.crossterm_source = eM
 end

function update_cross_term!(
    ct::InjectiveCrossTerm, eq::Conservation{Mass}, 
    target_storage, source_storage, 
    target_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    source_model::SimulationModel{<:Any, TS, <:Any, <:Any}, 
    target, source, dt
    ) where {TS <: ActiveMaterial, SS <:ElectrolyteModel}

    activematerial = TS
    electrolyte = SS 
    phi_e = target_storage.state.Phi[ct.impact.target]
    phi_a = source_storage.state.Phi[ct.impact.source]  
    ocd = source_storage.state.Ocd[ct.impact.source]
    R = source_storage.state.ReactionRateConst[ct.impact.source]
    c_e = source_storage.state.C[ct.impact.source]
    c_a = target_storage.state.C[ct.impact.target]

    eE, eM = sourceElectricMaterial(
        phi_a,c_a,R,ocd,
        value.(phi_e),value.(c_e),
        activematerial,electrolyte  
    )

    ct.crossterm_source = eM

    eE, eM = sourceElectricMaterial(
        value.(phi_a),value.(c_a),value.(R),value.(ocd),
        phi_e, c_e,
        activematerial,electrolyte  
    )
    
    ct.crossterm_target = eM

end
