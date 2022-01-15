abstract type AbstractCompositionalViscosity <: PhaseVariables end

struct LBCViscosities <: AbstractCompositionalViscosity end

@terv_secondary function update_as_secondary!(mu, m::LBCViscosities, model::SimulationModel{D, S}, param, Pressure, Temperature, FlashResults) where {D, S<:CompositionalSystem}
    eos = model.system.equation_of_state
    n = size(mu, 2)
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in 1:n
        p = Pressure[i]
        T = Temperature[1, i]
        mu[1, i], mu[2, i] = lbc_viscosities(eos, p, T, FlashResults[i])
    end
end

