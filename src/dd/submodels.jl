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


function submodel(model::MultiModel, mp::SimpleMultiModelPartition, index; kwarg...)
    p = main_partition(mp)
    main = mp.main_symbol
    submodels = model.models
    new_submodels = Dict()
    # First deal with main
    main_submodel = submodel(submodels[main], p, index)
    M = main_submodel.domain.global_map

    for k in keys(submodels)
        if k == main
            new_submodels[main] = main_submodel
        elseif mp.partition[k] == index
            @info index mp.partition[k]
            # Include the whole model
            # TODO: Renumber
            m = deepcopy(submodels[k])
            d = m.domain
            if isa(d, WellControllerDomain)
                # Need to control a single well for this to work
                @assert length(d.well_symbols) == 1
            elseif isa(m.domain.grid, WellGrid)
                perf = m.domain.grid.perforations.reservoir
                for i in eachindex(perf)
                    new_index = local_cell(perf[i], M)
                    perf[i] = new_index
                end
            else
                error("Not yet supported: $k")
            end
        end
    end
    # TODO: Groups, reduction, etc.
    sm = convert_to_immutable_storage(new_submodels)
    return MultiModel(sm)
end

