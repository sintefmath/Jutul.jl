function map_X_to_Y(F, X, model, parameters_map, state0_map, cache)
    has_state0 = !ismissing(state0_map)
    case = F(X, missing)
    N_prm = Jutul.vectorized_length(model, parameters_map)
    if has_state0
        N_s0 = Jutul.vectorized_length(model, state0_map)
    else
        N = N_prm
    end
    N = N_prm + N_s0
    T = eltype(X)
    if !haskey(cache, T)
        cache[T] = zeros(T, N)
    end
    Y = cache[T]
    resize!(Y, N)
    Y_prm = view(Y, 1:N_prm)
    vectorize_variables!(Y_prm, model, case.parameters, parameters_map)
    if has_state0
        Y_s0 = view(Y, (N_prm+1):(N_prm+N_s0))
        vectorize_variables!(Y_s0, model, case.state0, state0_map)
    end
    return Y
end

function setup_from_vectorized(Y, case, parameters_map, state0_map)
    has_state0 = !ismissing(state0_map)
    N_prm = Jutul.vectorized_length(case.model, parameters_map)
    if has_state0
        N_s0 = Jutul.vectorized_length(case.model, state0_map)
    else
        N = N_prm
    end
    N = N_prm + N_s0
    @assert length(Y) == N "Length of Y ($(length(Y))) does not match expected length ($N)."
    Y_prm = view(Y, 1:N_prm)
    devectorize_variables!(case.parameters, case.model, Y_prm, parameters_map)
    if has_state0
        Y_s0 = view(Y, (N_prm+1):(N_prm+N_s0))
        devectorize_variables!(case.state0, case.model, Y_s0, state0_map)
    end
    return case
end
