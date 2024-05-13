export vectorize_variables, vectorize_variables!, devectorize_variables!


function vectorize_variables(model, state_or_prm, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    n = vectorized_length(model, mapper)
    V = zeros(n)
    return vectorize_variables!(V, model, state_or_prm, mapper, config = config)
end

function vectorized_length(model, mapper)
    sum(x -> x.n, values(mapper), init = 0)
end

function vectorize_variables!(V, model, state_or_prm, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        F = opt_scaler_function(config, k, inv = false)
        if isnothing(config)
            c = nothing
        else
            c = config[k]
        end
        vectorize_variable!(V, state_or_prm, k, v, F, config = c)
    end
    return V
end

function vectorize_variable!(V, state, k, info, F; config = nothing)
    (; n, offset) = info
    state_val = state[k]
    lumping = get_lumping(config)
    if isnothing(lumping)
        @assert length(state_val) == n "Expected field $k to have length $n, was $(length(state_val))"
        if state_val isa AbstractVector
            for i in 1:n
                V[offset+i] = F(state_val[i])
            end
        else
            l, m = size(state_val)
            ctr = 1
            for i in 1:l
                for j in 1:m
                    V[offset+ctr] = F(state_val[i, j])
                    ctr += 1
                end
            end
        end
    else
        @assert length(state_val) == length(lumping)
        for i in 1:maximum(lumping)
            # Take the first lumped value as they must be equal by assumption
            ix = findfirst(isequal(i), lumping)
            V[offset+i] = state_val[ix]
        end
    end
end

function devectorize_variables!(state_or_prm, model, V, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        if isnothing(config)
            c = nothing
        else
            c = config[k]
        end
        F = opt_scaler_function(config, k, inv = true)
        devectorize_variable!(state_or_prm, V, k, v, F, config = c)
    end
    return state_or_prm
end

function devectorize_variable!(state, V, k, info, F_inv; config = c)
    (; n, offset) = info
    state_val = state[k]
    lumping = get_lumping(config)
    if isnothing(lumping)
        @assert length(state_val) == n "Expected field $k to have length $n, was $(length(state_val))"
        if state_val isa AbstractVector
            for i in 1:n
                state_val[i] = F_inv(V[offset+i])
            end
        else
            l, m = size(state_val)
            ctr = 1
            for i in 1:l
                for j in 1:m
                    state_val[i, j] = F_inv(V[offset+ctr])
                    ctr += 1
                end
            end
        end
    else
        for (i, lump) in enumerate(lumping)
            state_val[i] = F_inv(V[offset+lump])
        end
    end
end

function get_lumping(config::Nothing)
    return nothing
end

function get_lumping(config::AbstractDict)
    return config[:lumping]
end

get_mapper_internal(model, type_or_map::Symbol) = first(variable_mapper(model, type_or_map))
get_mapper_internal(model, type_or_map) = type_or_map

function vectorize_data_domain(d::DataDomain)
    n = 0
    for (k, val_e_pair) in pairs(d.data)
        val, e = val_e_pair
        if eltype(val)<:AbstractFloat
            n += length(val)
        end
    end
    x = zeros(n)
    return vectorize_data_domain!(x, d::DataDomain)
end

function vectorize_data_domain!(x, d::DataDomain)
    offset = 0
    for (k, val_e_pair) in pairs(d.data)
        val, e = val_e_pair
        if eltype(val)<:AbstractFloat
            n = length(val)
            for i in 1:n
                x[offset+i] = val[i]
            end
            offset += n
        end
    end
    return x
end

function devectorize_data_domain(domain::DataDomain{R, E, D}, x::Vector{T}) where {R, E, D, T}
    r = physical_representation(domain)
    e = deepcopy(domain.entities)
    d = similar(domain.data)
    newd = DataDomain{R, E, D}(r, e, d)
    for (k, val_e_pair) in pairs(domain)
        val, e = val_e_pair
        if eltype(val)<:AbstractFloat
            sz = size(val)
            newd[k, e] = zeros(T, sz)
        else
            newd[k, e] = copy(val)
        end
    end
    @assert keys(newd) == keys(d)
    return devectorize_data_domain!(newd, x)
end

function devectorize_data_domain!(d::DataDomain, x::Vector{T}) where T
    offset = 0
    for (k, val_e_pair) in pairs(d.data)
        val, e = val_e_pair
        if eltype(val)<:AbstractFloat
            n = length(val)
            if eltype(val) == T
                for i in 1:n
                    val[i] = x[offset+i]
                end
            else
                val = reshape(x[(offset+1):(offset+n)], size(val))
                d[k, e] = val
            end
            offset += n
        end
    end
    return d
end

function data_domain_to_parameters_gradient(model0)
    function get_ad_local(x::ST.ADval, rng, dim, n_total)
        v = x.val
        I0, V0 = findnz(ST.deriv(x))
        grad = SparseVector(n_total, I0, V0)[rng]
        if length(dim) == 2
            I, V = findnz(grad)
            bz, n = dim
            row = similar(I)
            col = similar(I)
            for (i, spos) in enumerate(I)
                row[i] = mod(spos-1, bz)+1
                col[i] = div(spos-1, bz)+1
            end
            grad = sparse(row, col, V, bz, n)
        else
            @assert length(dim) == 1
        end
        return grad
        # return (value = v, gradient = grad)
    end
    model = deepcopy(model0)
    d = model.data_domain
    x = vectorize_data_domain(d)
    n_total = length(x)
    x_ad = ST.create_advec(x)
    devectorize_data_domain!(d, x_ad)
    prm = setup_parameters(model)
    output_prm = Dict{Symbol, Any}()
    for (prm_name, prm_val) in pairs(prm)
        subprm = Dict{Symbol, Any}()
        offset = 0
        for (k, val_e_pair) in pairs(model0.data_domain.data)
            val, e = val_e_pair
            val_ad = d[k, e]
            if eltype(val)<:AbstractFloat
                n = length(val)
                rng = (1+offset):(offset+n)
                subprm[k] = map(
                    x -> get_ad_local(x, rng, size(val), n_total),
                    prm_val
                    )
                offset += n
            end
        end
        output_prm[prm_name] = subprm
    end
    return output_prm
end
