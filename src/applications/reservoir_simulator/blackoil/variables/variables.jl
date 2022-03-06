Base.@kwdef struct GasMassFraction <: ScalarVariable
    dz_max = 0.1
end

maximum_value(::GasMassFraction) = 1.0
minimum_value(::GasMassFraction) = 0.0
absolute_increment_limit(s::GasMassFraction) = s.dz_max

struct BlackOilPhaseState <: ScalarVariable

end

default_value(model, ::BlackOilPhaseState) = OilAndGas
initialize_secondary_variable_ad!(state, model, var::BlackOilPhaseState, arg...; kwarg...) = state

@terv_secondary function update_as_secondary!(phase_state, m::BlackOilPhaseState, model::SimulationModel{D, S}, param, Pressure, GasMassFraction) where {D, S<:BlackOilSystem}
    pvt = model.system.pvt
    for i in eachindex(phase_state)
        p = Pressure[i]
        z_g = GasMassFraction[i]
        if max_dissolved_gas_fraction(pvt, p) > z_g
            phase_state[i] = OilAndGas
        else
            phase_state[i] = OilOnly
        end
    end
end

struct Rs <: ScalarVariable end

@terv_secondary function update_as_secondary!(rs, m::Rs, model::SimulationModel{D, S}, param, PhaseState, Pressure, GasMassFraction) where {D, S<:BlackOilSystem}
    pvt = model.system.pvt
    for i in eachindex(phase_state)
        if PhaseState[i] == OilAndGas
            z_g = max_dissolved_gas_fraction(pvt, p)
        else
            z_g = GasMassFraction[i]
        end
        rs[i] = ρ_os*z_g/(ρ_gs*(1-z_g))
    end
end


# @terv_secondary function update_as_secondary!(rho, m::DeckDensity, model::SimulationModel{D, S}, param, Pressure, Rs) where {D, S<:BlackOilSystem}
#     sys = model.system
#     rhos = param[:reference_densities]
#     pvt, reg = ρ.pvt, ρ.regions

#     # eos = sys.equation_of_state
#     n = size(rho, 2)
#     for i = 1:n
#         rs = Rs[i]
#         p = Pressure[i]
#         rho[1, i] = oil_density(opvt, p, rs)
#         rho[2, i] = gas_density(gpvt, p)
#     end
# end

# @terv_secondary function update_as_secondary!(mu, μ::DeckViscosity, model::SimulationModel{D, S}, param, Pressure, Rs) where {D, S<:BlackOilSystem}
#     pvt, reg = μ.pvt, μ.regions
#     tb = thread_batch(model.context)
#     nph, nc = size(mu)
#     for ph in 1:nph
#         pvt_ph = pvt[ph]
#         @batch minbatch = tb for i in 1:nc
#             p = Pressure[i]
#             @inbounds mu[ph, i] = viscosity(pvt_ph, reg, p, i)
#         end
#     end
# end
