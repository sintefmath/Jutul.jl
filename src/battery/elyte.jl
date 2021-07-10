using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct TPDGrad{T} <: KGrad{T} end
struct ChemicalCurrent <: ScalarVariable end
struct TotalCurrent <: ScalarVariable end
struct ChargeCarrierFlux <: ScalarVariable end
# Is it necesessary with a new struxt for all these?
struct Conductivity <: ScalarVariable end
struct Diffusivity <: ScalarVariable end
struct DmuDc <: ScalarVariable end

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


function select_secondary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPDGrad_C] = TPDGrad{C}()
    S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:Conductivity] = Conductivity()
    S[:DmuDc] = DmuDc()
    S[:Diffusivity] = Diffusivity()
    S[:TotalCurrent] = TotalCurrent()
    S[:ChargeCarrierFlux] = ChargeCarrierFlux()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = EnergyAcc()

    t = 1; z = 1
    S[:t] = ConstantVariables([t])
    S[:z] = ConstantVariables([z])
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :EnergyAcc]
end

@terv_secondary function update_as_secondary!(
    dmudc, sv::DmuDc, model, param, T, C
    )
    # TODO: Find a better way to hadle constants
    R = 1 # Gas constant i proper units
    @tullio dmudc[i] = R * (T[i] / C[i])
end

function cond(
    C, T, model::SimulationModel{D, S, F, Con},
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    return 4. + 3.5*C + (2.)*T + 0.1 * C * T # Arbitrary for now
end

@terv_secondary function update_as_secondary!(
    κ, sv::Conductivity, model, param, T, C
    )
    @tullio κ[i] = cond(T[i], C[i], model)
end

function diffusivity(
    c, T, model::SimulationModel{D, S, F, Con},
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    cnst = [[-4.43, -54] [-0.22, 0.0 ]]

    Tgi = [229, 5.0]

    # Diffusion coefficient, [m^2 s^-1]
    return (
        1e-4 * 10 ^ ( ( cnst[1,1] + 
        cnst[1,2] / ( T - Tgi[1] - Tgi[2] * c * 1e-3) + 
        cnst[2,1] * c * 1e-3) )
        )
    end

@terv_secondary function update_as_secondary!(
    D, sv::Diffusivity, model, param, C, T
    )
    @tullio D[i] = diffusivity(C[i], T[i], model)
end


    
@terv_secondary function update_as_secondary!(
    kgrad_phi, tv::TPkGrad{Phi}, model::SimulationModel{D, S, F, C}, param, 
    Phi, Conductivity
    ) where {D, S <: ElectroChemicalComponent, F, C}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio kgrad_phi[i] = half_face_two_point_kgrad(conn_data[i], Phi, Conductivity)
end


function get_alpha(
    model::SimulationModel{D, S, F, Con}
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    return ones(number_of_units(model, C()))
end

@terv_secondary function update_as_secondary!(
    pot, tv::TPkGrad{C}, model::SimulationModel{D, S, F, Con}, param, C,
    Conductivity, DmuDc, t, z
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    Far = 1 # Faradays constant in proper units
    # Should this be its own variable
    @tullio k[i] := Conductivity[i] * DmuDc[i] * t / (Far * z)
    @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], C, k)
end

@terv_secondary function update_as_secondary!(
    DGrad_C, tv::TPDGrad{C}, model::SimulationModel{Dom, S, F, Con}, 
    param, C, D 
    ) where {Dom, S <: ElectroChemicalComponent, F, Con}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio DGrad_C[i] = half_face_two_point_kgrad(conn_data[i], C, D)
end


@terv_secondary function update_as_secondary!(
    j, tv::TotalCurrent, model, param, 
    TPkGrad_C, TPkGrad_Phi
    )
    @tullio j[i] =  -TPkGrad_C[i] + TPkGrad_Phi[i]
end

@terv_secondary function update_as_secondary!(
    N, tv::ChargeCarrierFlux, model, param, 
    TPDGrad_C, TotalCurrent, t, z
    )
    a = t / (F * z)
    @tullio N[i] =  -TPDGrad_C[i] +  a * TotalCurrent[i]
end


function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{MassAcc}) where {D, S <: Electrolyte, F, Con}
    return storage.state.TotalCurrent
end
