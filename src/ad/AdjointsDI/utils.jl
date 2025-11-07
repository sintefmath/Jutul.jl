function setup_vectorize_nested(data, active = missing; kwarg...)
    meta = (
        offsets = Int[1],
        names = [],
        dims = [],
        types = Symbol[]
    )
    setup_vectorize_nested!(meta, data, active; kwarg...)
    return meta
end

function setup_vectorize_nested!(meta, data, active = missing;
        header = [],
        active_type = Float64,
        multipliers = missing
    )
    function name_is_active(n)
        act = false
        if ismissing(active)
            act = true
        else
            for active_name in active
                this_name_act = true
                for i in 1:min(length(active_name), length(n))
                    if n[i] != active_name[i]
                        this_name_act = false
                        break
                    end
                end
                if this_name_act
                    act = true
                    break
                end
            end
        end
        return act
    end
    for (k, v) in pairs(data)
        if v isa AbstractDict || v isa JutulStorage
            subheader = copy(header)
            push!(subheader, k)
            setup_vectorize_nested!(meta, v, active; active_type = active_type, header = subheader)
        elseif v isa Union{active_type, AbstractArray{<:active_type}}
            name = copy(header)
            push!(name, k)
            if name_is_active(name)
                if v isa AbstractArray
                    d = size(v)
                    num = length(v)
                else
                    d = nothing
                    num = 1
                end
                push!(meta.offsets, meta.offsets[end] + num)
                push!(meta.names, name)
                push!(meta.types, :value)
                push!(meta.dims, d)
            end
        else
            continue
        end
    end
    if !ismissing(multipliers)
        for (name, mult) in pairs(multipliers)
            v = mult.value
            if v isa AbstractArray
                d = size(v)
                num = length(v)
            else
                d = nothing
                num = 1
            end
            @info "????" name
            push!(meta.names, name)
            push!(meta.offsets, meta.offsets[end] + num)
            push!(meta.types, :multiplier)
            push!(meta.dims, d)
        end
    end
end

function vectorize_nested(data; kwarg...)
    return vectorize_nested!(missing, data; kwarg...)
end

function vectorize_nested!(x, data; setup = missing, active = missing, active_type = Float64, multipliers = missing)
    function get_subdict(name_list::Vector)
        d = data
        for name in name_list[1:end-1]
            d = d[name]
        end
        return d
    end
    if !ismissing(active) && !ismissing(setup)
        throw(ArgumentError("Both setup and active cannot be provided."))
    end
    if ismissing(setup)
        setup = setup_vectorize_nested(data, active, active_type = active_type, multipliers = multipliers)
    end
    n = setup.offsets[end]-1
    if ismissing(x)
        x = zeros(n)
    end
    for (i, name) in enumerate(setup.names)
        vtype = setup.types[i]
        start = setup.offsets[i]
        stop = setup.offsets[i+1]-1
        if vtype == :value
            d = get_subdict(name)
            lastname = name[end]
            v = d[lastname]
        else
            v = multipliers[name].value
        end
        if v isa AbstractArray
            subx = view(x, start:stop)
            for i in eachindex(subx, v)
                subx[i] = v[i]
            end
        else
            @assert start == stop "Expected start=$start=$stop=stop for scalar $v"
            x[start] = v
        end
    end
    return (x, setup)
end

function devectorize_nested(x, setup)
    data = Dict()
    return devectorize_nested!(data, x, setup)
end

function devectorize_nested!(data, x, setup)
    function get_subdict(name_list::Vector)
        d = data
        for name in name_list[1:end-1]
            if !haskey(d, name)
                d[name] = Dict()
            end
            d = d[name]
        end
        return d
    end

    for (i, name) in enumerate(setup.names)
        start = setup.offsets[i]
        stop = setup.offsets[i+1]-1
        dims = setup.dims[i]
        d = get_subdict(name)
        lastname = name[end]
        if isnothing(dims)
            @assert start == stop "Expected start=$start=$stop=stop for scalar $name"
            d[lastname] = x[start]
        else
            if haskey(d, lastname) && eltype(d[lastname]) == eltype(x)
                subx = view(x, start:stop)
                v = d[lastname]
                for i in eachindex(subx, v)
                    v[i] = subx[i]
                end
            else
                d[lastname] = reshape(x[start:stop], dims)
            end
        end
    end
    return data
end
