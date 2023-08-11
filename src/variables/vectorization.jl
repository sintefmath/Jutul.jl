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
