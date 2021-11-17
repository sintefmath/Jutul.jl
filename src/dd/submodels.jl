export submodel
function submodel(model::SimulationModel, partition::AbstractDomainPartition, index; kwarg...)
    p_i = entity_subset(partition, index)
    return submodel(model, p_i; kwarg...)
end

function submodel(model::SimulationModel, p_i::AbstractVector; kwarg...)
    domain = model.domain
    sys = model.system
    c, f = model.context, model.formulation
    d_l = subdomain(domain, p_i, entity = Cells(); kwarg...)
    new_model = SimulationModel(d_l, sys, context = c, formulation = f)
    M = global_map(new_model.domain)
    function transfer_vars!(new, old)
        for k in keys(old)
            new[k] = subvariable(old[k], M)
        end
    end
    transfer_vars!(new_model.primary_variables, model.primary_variables)
    transfer_vars!(new_model.secondary_variables, model.secondary_variables)
    return new_model
end


function submodel(model::MultiModel, mp::SimpleMultiModelPartition, index; kwarg...)
    p = main_partition(mp)
    main = mp.main_symbol
    submodels = model.models
    # Preserve order of models (at least of those that will be included)
    new_submodels = OrderedDict()
    # First deal with main
    main_submodel = submodel(submodels[main], p, index; kwarg...)
    M = main_submodel.domain.global_map
    groups_0 = model.groups
    has_groups = !isnothing(groups_0)
    if has_groups
        groups = Vector{Integer}()
    end

    for (i, k) in enumerate(keys(submodels))
        if k == main
            new_submodels[main] = main_submodel
        elseif mp.partition[k] == index
            # Include the whole model, somewhat of a hack for wells
            # TODO: Renumber
            m = deepcopy(submodels[k])
            d = m.domain
            if isa(d, WellControllerDomain)
                # Need to control a single well for this to work
                @assert length(d.well_symbols) == 1
            elseif isa(m.domain.grid, WellGrid)
                perf = m.domain.grid.perforations.reservoir
                for i in eachindex(perf)
                    c_l = local_cell(perf[i], M)
                    @assert !isnothing(c_l)
                    # new_index = interior_cell(c_l, M)
                    perf[i] = c_l
                end
            else
                error("Not yet supported: $k")
            end
            new_submodels[k] = m
        else
            # @debug "Skipping submodel #$i: $k, not found in local partition."
            continue
        end
        # We didn't continue, so we can append the group
        if has_groups
            push!(groups, groups_0[i])
        end
    end
    if !has_groups || length(groups) == 1
        groups = nothing
        reduction = nothing
        ctx = submodels[1].context
    else
        reduction = model.reduction
        ctx = model.context
    end
    # TODO: Renumber groups in case only one group persists.
    sm = convert_to_immutable_storage(new_submodels)
    return MultiModel(sm, groups = groups, reduction = reduction, context = ctx)
end

subvariable(var, map) = var
