struct TwoPhaseCompositionalDensities <: PhaseMassDensities
end

@terv_secondary function update_as_secondary!(rho, m::TwoPhaseCompositionalDensities, model::SimulationModel{D, S}, param, Pressure, Temperature, FlashResults) where {D, S<:CompositionalSystem}
    eos = model.system.equation_of_state
    n = size(rho, 2)
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in 1:n
        p = Pressure[i]
        T = Temperature[1, i]
        rho[1, i], rho[2, i] = mass_densities(eos, p, T, FlashResults[i])
    end
end
