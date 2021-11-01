export submodel
function submodel(model::SimulationModel, partition::AbstractDomainPartition, index; kwarg...)
    p_i = entity_subset(partition, index)
    return submodel(model, p_i; kwarg...)
end

function submodel(model::SimulationModel, p_i::AbstractVector; kwarg...)
    domain = model.domain
    sys = model.system
    d_l = subdomain(domain, p_i, entity = Cells(); kwarg...)
    return SimulationModel(d_l, sys)
end

