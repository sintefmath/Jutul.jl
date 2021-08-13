using Terv
export CurrentCollector

struct CurrentCollector <: ElectroChemicalComponent end

function minimum_output_variables(
    system::CurrentCollector, primary_variables
    )
    [:ChargeAcc, :TPkGrad_Phi]
end

function select_primary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:Phi] = Phi()
end

function select_secondary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:ChargeAcc] = ChargeAcc()
    S[:Conductivity] = Conductivity()
end


function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
end

function update_cross_term!(ct::InjectiveCrossTerm, eq::Conservation{ChargeAcc}, 
    target_storage, source_storage, 
    target_model, source_model, 
    target, source, dt)
    phi_t = target_storage.state.Phi[ct.impact.target]
    phi_s = source_storage.state.Phi[ct.impact.source]
    function interfaceflux!(src, phi_1,phi_2)
        for i in eachindex(phi_1)
            src[i] = (phi_2[i] - phi_1[i])
            src[i] *= 1e7
        end
        # src = phi_2 -phi_1 does not work
    end
    interfaceflux!(ct.crossterm_source,phi_s,value.(phi_t))
    interfaceflux!(ct.crossterm_target,value.(phi_s),phi_t)
    #@. ct.crossterm_source = interfaceflux(phi_s,value.(phi_t))
    #@. ct.crossterm_target = interfaceflux(value.(phi_s), phi_t)
    #error("Cross term must be specialized for your equation and models. Did not understand how to specialize $target ($(typeof(target_model))) to $source ($(typeof(source_model)))")
end