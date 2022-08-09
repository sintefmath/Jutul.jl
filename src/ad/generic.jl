function generic_cache_declare_pattern(cache::GenericAutoDiffCache, entity_indices = 1:number_of_entities(cache))
    J = cache.variables
    I = similar(J)
    n = number_of_entities(cache)
    for i in 1:n
        e_i = entity_indices[i]
        for j in vrange(cache, i)
            I[j] = e_i
        end
    end
    return (I, J)
end

number_of_partials(::Type{ForwardDiff.Dual{T, V, N}}) where {T,V,N} = N

entity(::GenericAutoDiffCache{<:Any, E}) where E = E
equations_per_entity(::GenericAutoDiffCache{N}) where N = N
number_of_entities(c::GenericAutoDiffCache) = length(c.vpos)-1
number_of_partials(c::GenericAutoDiffCache{N, E, ∂T}) where {N, E, ∂T} = number_of_partials(∂T)

vrange(c::GenericAutoDiffCache, i) = c.vpos[i]:(c.vpos[i+1]-1)
get_entries(c::GenericAutoDiffCache) = c.entries

@inline function get_jacobian_pos(c::GenericAutoDiffCache, index, eqNo, partial_index, pos = c.jacobian_positions)
    np = number_of_partials(c)
    @inbounds pos[(eqNo-1)*np + partial_index, index]
end

function diagonal_view(cache::GenericAutoDiffCache)
    dpos = cache.diagonal_positions
    @assert !isnothing(dpos)
    return view(cache.entries, :, dpos)
end

function fill_equation_entries!(nz, r, model, cache::GenericAutoDiffCache)
    nu, ne, np = ad_dims(cache)
    entries = cache.entries
    tb = minbatch(model.context)
    dpos = cache.diagonal_positions
    is_diag_pos(i, jno, dpos) = dpos[jno] == i
    is_diag_pos(i, jno, ::Nothing) = jno == 1

    get_diagonal_pos(dpos, i) = dpos[i]
    get_diagonal_pos(dpos::Nothing, i) = i
    # @batch minbatch = tb for i in 1:nu
    if isnothing(dpos)
        # We don't have diagonals, just fill inn residual whenever
        diag_index = dpos[i]
        for (jno, j) in enumerate(vrange(cache, i))
            fill_residual = jno == 1
            for e in 1:ne
                a = entries[e, j]
                if fill_residual
                    insert_residual_value(r, i, e, a.value)
                end
                for d = 1:np
                    update_jacobian_entry!(nz, cache, j, e, d, a.partials[d])
                end
            end
        end
    else
        # Diagonal value might differ due to source terms, be careful
        for i in 1:nu
            diag_index = dpos[i]
            for j in vrange(cache, i)
                fill_residual = j == diag_index
                for e in 1:ne
                    a = entries[e, j]
                    if fill_residual
                        insert_residual_value(r, i, e, a.value)
                    end
                    for d = 1:np
                        update_jacobian_entry!(nz, cache, j, e, d, a.partials[d])
                    end
                end
            end
        end
    end
end
