export submodel
function submodel(model::SimulationModel, partition::AbstractDomainPartition, index; kwarg...)
    p_i = entity_subset(partition, index)
    return submodel(model, p_i; kwarg...)
end

function submodel(model::SimulationModel, p_i::AbstractVector; context = model.context, minbatch = nothing, kwarg...)
    domain = model.domain
    sys = model.system
    f = model.formulation
    if !isnothing(minbatch)
        T = typeof(context)
        # Hope the constructor fits.
        context = T(matrix_layout = context.matrix_layout, minbatch = minbatch)
    end
    d_l = subdomain(domain, p_i, entity = Cells(); kwarg...)
    new_model = SimulationModel(d_l, sys, context = context, formulation = f)
    M = global_map(new_model.domain)
    function transfer_vars!(new, old)
        for k in keys(new)
            delete!(new, k)
        end
        for k in keys(old)
            new[k] = subvariable(old[k], M)
        end
    end
    transfer_vars!(new_model.primary_variables, model.primary_variables)
    transfer_vars!(new_model.secondary_variables, model.secondary_variables)
    transfer_vars!(new_model.parameters, model.parameters)

    new_data_domain = new_model.data_domain
    old_data_domain = model.data_domain
    transfer_data_domain_values!(new_data_domain, old_data_domain, global_map(new_model))
    return new_model
end

function transfer_data_domain_values!(new_data_domain::DataDomain, old_data_domain::DataDomain, m::FiniteVolumeGlobalMap)
    transfer_data_domain_values!(new_data_domain, old_data_domain, Cells(), m.cells)
    transfer_data_domain_values!(new_data_domain, old_data_domain, Faces(), m.faces)
    return new_data_domain
end

function transfer_data_domain_values!(new_data_domain::DataDomain, old_data_domain::DataDomain, m::TrivialGlobalMap)
    for (k, v) in pairs(old_data_domain)
        e = associated_entity(old_data_domain, k)
        setindex!(new_data_domain, copy(first(v)), k, e)
    end
    return new_data_domain
end

function transfer_data_domain_values!(new_data_domain::DataDomain, old_data_domain::DataDomain, e::JutulEntity, e_ix)
    for (k, v_pair) in pairs(old_data_domain)
        v, e_i = v_pair
        if e_i != e
            continue
        end
        if v isa AbstractVector
            subv = v[e_ix]
        else
            subv = v[:, e_ix]
        end
        setindex!(new_data_domain, subv, k, e)
    end
    return new_data_domain
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
            pr = physical_representation(m.domain)
            if hasproperty(d, :well_symbols)
                # Need to control a single well for this to work
                @assert length(d.well_symbols) == 1
            elseif hasproperty(pr, :perforations)
                # Really hacky, this should live in JutulDarcy.
                perf = physical_representation(m.domain).perforations.reservoir
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
    # Cross terms...
    mk = keys(new_submodels)
    sm = convert_to_immutable_storage(new_submodels)
    new_model = MultiModel(sm, groups = groups, reduction = reduction, context = ctx)

    for ctp in model.cross_terms
        (; target, source) = ctp
        if target in mk && source in mk
            sub_ctp = subcrossterm_pair(ctp, new_model, mp)
            add_cross_term!(new_model, sub_ctp)
        end
    end
    return new_model
end

"""
    subvariable(var, map)

Get subvariable of Jutul variable
"""
function subvariable(var, map)
    if hasproperty(var, :regions)
        @warn "Default subvariable called for $(typeof(var)) that contains regions property. Potential missing interface specialization."
    end
    return var
end

function subvariable(var::Pair, map)
    label, var = var
    return Pair(label, subvariable(var, map))
end

partition_variable_slice(v::AbstractVector, partition) = v[partition]
partition_variable_slice(v::AbstractMatrix, partition) = v[:, partition]
partition_variable_slice(v, partition) = v
