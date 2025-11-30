export vectorize_variables, vectorize_variables!, devectorize_variables!, devectorize_state_and_parameters!


function vectorize_variables(model, state_or_prm, type_or_map = :primary; config = nothing, T = Float64)
    mapper = get_mapper_internal(model, type_or_map)
    n = vectorized_length(model, mapper)
    V = zeros(T, n)
    return vectorize_variables!(V, model, state_or_prm, mapper, config = config)
end

function vectorized_length(model, mapper)
    sum(x -> x.n_x, values(mapper), init = 0)
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
        vardef = model[k]
        vectorize_variable!(V, state_or_prm, k, v, F, model, vardef, config = c)
    end
    return V
end

function vectorize_variable!(V, state, k, info, F, model, vardef::JutulVariables; config = nothing)
    (; n_full, n_x, offset_full, offset_x) = info
    state_val = state[k]
    lumping = get_lumping(config)
    if isnothing(lumping)
        @assert n_full == n_x
        @assert length(state_val) == n_x "Expected field $k to have length $n_x, was $(length(state_val))"
        local_offset = 0
        if state_val isa AbstractVector
            iterator = 1:n_x
        else
            iterator = axes(state_val, 1)
        end
        for i in iterator
            added = vectorize_variable_values!(V, i, offset_x + local_offset, F, state_val, model, vardef)
            local_offset += added
        end
    else
        lumping::AbstractVector
        if state_val isa AbstractVector
            @assert length(state_val) == length(lumping) "Lumping must be given as a vector with one value per column for vector, was $(length(lumping))"
        else
            @assert size(state_val, 2) == length(lumping) "Lumping must be given as a vector with one value per column for matrix, was $(length(lumping))"
        end
        iterator = 1:maximum(lumping)
        local_offset = 0
        for i in iterator
            ix = findfirst(isequal(i), lumping)
            added = vectorize_variable_values!(V, ix, offset_x + local_offset, F, state_val, model, vardef)
            local_offset += added
        end
    end
    return V
end

function vectorize_variable_values!(dest, idx, dest_offset, F, state_val::AbstractArray, model::JutulModel, variable_def::JutulVariables)
    el = scalarize_variable(model, state_val, variable_def, idx, numeric = true)
    dest[dest_offset + idx] = F(el)
    return 1
end

function vectorize_variable_values!(dest, idx, dest_offset, F, state_val::AbstractVector, model::JutulModel, variable_def::JutulVariables)
    el = scalarize_variable(model, state_val, variable_def, idx, numeric = true)
    m = length(el)
    for j in 1:m
        dest[dest_offset + j] = F(el[m])
    end
    return m
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

function devectorize_variable!(state, V, k, info, F_inv; config = nothing)
    (; n_full, n_x, offset_full, offset_x) = info
    state_val = state[k]
    T = eltype(V)
    if eltype(state_val) != T
        state_val = similar(state_val, T)
        state[k] = state_val
    end
    lumping = get_lumping(config)
    devectorize_variable_inner!(state_val, V, k, F_inv, lumping, n_full, n_x, offset_full, offset_x)
    return state
end

function devectorize_variable_inner!(state_val, V, k, F_inv, lumping, n_full, n_x, offset_full, offset_x)
    if isnothing(lumping)
        @assert n_full == n_x
        @assert length(state_val) == n_full "Expected field $k to have length $n_full, was $(length(state_val))"
        if state_val isa AbstractVector
            for i in 1:n_full
                state_val[i] = F_inv(V[offset_x+i])
            end
        else
            l, m = size(state_val)
            ctr = 1
            for i in 1:l
                for j in 1:m
                    state_val[i, j] = F_inv(V[offset_x+ctr])
                    ctr += 1
                end
            end
        end
    else
        if state_val isa AbstractVector
            for (i, lump) in enumerate(lumping)
                state_val[i] = F_inv(V[offset_x+lump])
            end
        else
            lumping::AbstractVector
            m, ncell = size(state_val)
            nlump = length(lumping)
            @assert ncell == nlump "Lumping must be given as a vector with one value per column ($ncell) for matrix, was $nlump"
            for (i, lump) in enumerate(lumping)
                for j in 1:m
                    state_val[j, i] = F_inv(V[offset_x+(lump-1)*m+j])
                end
            end
        end
    end
end

function devectorize_state_and_parameters!(state, parameters, model, V, mapper, config)
    state_and_parameters = merge(state, parameters)
    devectorize_variables!(state_and_parameters, model, V, mapper; config=config)
    for k in keys(mapper)
        if haskey(state, k)
            state[k] = state_and_parameters[k]
        else
            parameters[k] = state_and_parameters[k]
        end
    end
    return (state, parameters)
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
    d = empty(domain.data)
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


"""
    parameters_jacobian_wrt_data_domain(model; copy = true, config = nothing)

Compute the (sparse) Jacobian of parameters with respect to data_domain values
(i.e. floating point values). Optionally, `config` can be passed to allow
`vectorize_variables` to only include a subset of the parameters.
"""
function parameters_jacobian_wrt_data_domain(model;
        copy = true,
        config = nothing,
        use_di = true,
        backend = default_di_backend(),
        prep = missing
    )
    data_domain = model.data_domain
    x = vectorize_data_domain(data_domain)
    if use_di
        if copy
            model = deepcopy(model)
        end
        function F(X)
            model_tmp = deepcopy(model)
            dd = model_tmp.data_domain
            devectorize_data_domain!(dd, X)
            prm = setup_parameters(model_tmp, perform_copy = false)
            return vectorize_variables(model_tmp, prm, :parameters, T = eltype(X), config = config)
        end
        if ismissing(prep)
            J = jacobian(F, backend, x)
        else
            J = jacobian(F, prep, backend, x)
        end
    else
        x_ad = ST.create_advec(x)
        devectorize_data_domain!(data_domain, x_ad)
        prm = setup_parameters(model, perform_copy = false)
        prm_flat = vectorize_variables(model, prm, :parameters, T = eltype(x_ad), config = config)
        n_parameters = length(prm_flat)
        n_data_domain_values = length(x)
        # This is Jacobian of parameters with respect to data_domain
        J = ST.jacobian(prm_flat, n_data_domain_values)
        @assert size(J) == (n_parameters, n_data_domain_values)
    end
    return J
end

function default_di_backend(; sparse = true)
    if sparse
        sparsity_detector = TracerLocalSparsityDetector(gradient_pattern_type=Set{Int})
        backend = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = sparsity_detector,
            coloring_algorithm = GreedyColoringAlgorithm(),
        )
    else
        backend = AutoForwardDiff()
    end
    return backend
end

"""
    data_domain_to_parameters_gradient(model, parameter_gradient; dp_dd = missing, config = nothing)

Make a data_domain copy that contains the gradient of some objective with
respect to the fields in the data_domain, assuming that the parameters were
initialized directly from the data_domain via (`setup_parameters`)[@ref].
"""
function data_domain_to_parameters_gradient(model, parameter_gradient; dp_dd = missing, config = nothing, kwarg...)
    if ismissing(dp_dd)
        dp_dd = parameters_jacobian_wrt_data_domain(model; copy = true, config = config, kwarg...)
    end
    do_dp = vectorize_variables(model, parameter_gradient, :parameters, config = config)
    # do/dd = do/dp * dp/dd
    # do_dp is a column vector, turn into row vector:
    # do_dd = do_dp'*dp_dd
    # if we want to output do_dd' (standard julia vector )we can rewrite
    # do_dd' = (do_dp'*dp_dd)' = dp_dd'*do_dp
    do_dd = dp_dd'*do_dp
    data_domain_with_gradients = deepcopy(model.data_domain)
    devectorize_data_domain!(data_domain_with_gradients, do_dd)
    return data_domain_with_gradients
end
