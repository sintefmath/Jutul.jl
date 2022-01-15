
@terv_secondary function update_as_secondary!(Sat, s::Saturations, model::SimulationModel{D, S}, param, FlashResults) where {D, S<:CompositionalSystem}
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in 1:size(Sat, 2)
        S_l, S_v = phase_saturations(FlashResults[i])
        Sat[1, i] = S_l
        Sat[2, i] = S_v
    end
end
