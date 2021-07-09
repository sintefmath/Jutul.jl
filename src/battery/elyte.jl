using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct ChemicalCurrent <: ScalarVariable end
struct TotalCurrent <: ScalarVariable end

function number_of_units(model, pv::TotalCurrent)
    """ Two fluxes per face """
    return 2*count_units(model.domain, Faces())
end

function select_equations_system!(
    eqs, domain, system::Electrolyte, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(
        ChargeAcc(), arg...; kwarg...
        )
    mass_cons = (arg...; kwarg...) -> Conservation(
        MassAcc(), arg...; kwarg...
        )
    energy_cons = (arg...; kwarg...) -> Conservation(
        EnergyAcc(), arg...; kwarg...
        )
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)

end

function select_primary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
    S[:T] = T()
end

# Should not use only flow type
function select_secondary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:TotalCurrent] = TotalCurrent()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = EnergyAcc()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :EnergyAcc]
end


function get_kappa(
    C, T, model::SimulationModel{D, S, F, Con},
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    return 4. + 3.5*C + (2.)*T + 0.1 * C * T # Arbitrary for now
end

@terv_secondary function update_as_secondary!(
    pot, tv::C, model::SimulationModel{D, S, F, Con}, param, C, T
    ) where {D, S <: ElyteTest, F, Con}
    mf = model.domain.discretizations.mi
    conn_data = mf.conn_data
    context = model.context
    # ?Her burde en kanskje ha noe pre allocated?
    @. κ = get_kappa(C, T, model) # Kanskje ikke så lurt med κ i kode...
    update_cell_neighbor_potential_cc!(
        pot, conn_data, C, context, kernel_compatibility(context), κ
        )
end

@terv_secondary function update_as_secondary!(
    j, tv::TotalCurrent, model, param, 
    TPkGrad_C, TPkGrad_Phi
    )
    # Should have one coefficient for each, probably
    @tullio j[i] =  TPkGrad_C[i] + TPkGrad_Phi[i]
end


function update_half_face_flux!(
    law::Conservation{MassAcc}, storage, model, dt, 
    flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    j = storage.state.TotalCurrent
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = j[i]
end
