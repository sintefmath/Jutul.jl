function active_keys(dopt::DictParameters)
    return keys(dopt.parameter_targets)
end

function inactive_keys(dopt::DictParameters)
    return setdiff(all_keys(dopt), active_keys(dopt))
end

function all_keys(dopt::DictParameters)
    return dopt.possible_targets
end

function limit_name(rel::Bool, is_max::Bool)
    if rel
        if is_max
            s = raw"rel_max"
        else
            s = raw"rel_min"
        end
    else
        if is_max
            s = raw"abs_max"
        else
            s = raw"abs_min"
        end
    end
    return s
end

function realize_limit(dopt::DictParameters, parameter_name; is_max::Bool)
    vals = get_nested_dict_value(dopt.parameters, parameter_name)
    lims = get_parameter_limits(dopt, parameter_name)
    return realize_limit(vals, lims, is_max = is_max)
end

function realize_limit(initial::Union{Number, Array}, lims::KeyLimits; is_max::Bool)
    if is_max
        l = realize_limit_inner(initial, lims.rel_max, lims.abs_max, is_max = true)
    else
        l = realize_limit_inner(initial, lims.rel_min, lims.abs_min, is_max = false)
    end
    return l
end

function limit_getindex(x::Number, I)
    return x
end

function limit_getindex(x::Array, I)
    return x[I]
end

function realize_limit_inner(initial::Array, rel, abs; is_max::Bool)
    out = similar(initial)
    for (i, v) in enumerate(initial)
        r = limit_getindex(rel, i)
        a = limit_getindex(abs, i)
        out[i] = realize_limit_inner(v, r, a, is_max = is_max)
    end
    return out
end

function realize_limit_inner(initial::Number, rel_lim::Number, abs_lim::Number; is_max::Bool)
    rel_delta = abs(initial*(rel_lim-1.0))
    if is_max
        l = min(abs_lim, initial + rel_delta)
        @assert initial <= l
    else
        l = max(abs_lim, initial - rel_delta)
        @assert initial >= l
    end
    return l
end

function realize_limits(dopt::DictParameters, parameter_name)
    lb = realize_limit(dopt, parameter_name, is_max = false)
    ub = realize_limit(dopt, parameter_name, is_max = true)
    return (min = lb, max = ub)
end

function realize_limits(dopt::DictParameters)
    lb = Float64[]
    ub = Float64[]
    for parameter_name in active_keys(dopt)
        lims = realize_limits(dopt, parameter_name)
        if lims.min isa Number
            lims.max::Number
            push!(lb, lims.min)
            push!(ub, lims.max)
        else
            for i in eachindex(lims.min, lims.max)
                push!(lb, lims.min[i])
                push!(ub, lims.max[i])
            end
        end
    end
    @assert length(lb) == length(ub)
    return (min = lb, max = ub)
end

function print_optimization_overview(dopt::DictParameters; io = Base.stdout, print_inactive = false)
    function avg(x)
        return x
    end

    function avg(x::AbstractArray)
        return sum(x)/length(x)
    end

    function format_value(x)
        return "$x (scalar)"
    end

    function format_value(x::AbstractArray)
        u = unique(x)
        N = length(x)
        if length(u) == 1
            return "$(only(u)) ($N values)"
        else
            a = avg(x)
            return "Avg. $a ($N values)"
        end
    end

    function print_table(subkeys, t, print_opt = true)
        pt = dopt.parameter_targets
        prm = dopt.parameters
        prm_opt = dopt.parameters_optimized
        is_optimized = !ismissing(prm_opt) && print_opt
        header = ["Name", "Initial value", "Bounds"]
        if is_optimized
            push!(header, "Optimized value")
            push!(header, "Change")
        end
        tab = Matrix{Any}(undef, length(subkeys), length(header))
        for (i, k) in enumerate(subkeys)
            v0 = get_nested_dict_value(prm, k)
            v0_avg = avg(v0)
            if haskey(pt, k)
                lims = realize_limits(dopt, k)
                limstr = "$(minimum(lims.min)) to $(maximum(lims.max))"
            else
                limstr = "(Not set)"
            end
            tab[i, 1] = join(k, ".")
            tab[i, 2] = format_value(v0)
            tab[i, 3] = limstr
            if is_optimized
                v = get_nested_dict_value(prm_opt, k)
                v_avg = avg(v)
                perc = round(100*(v_avg-v0_avg)/max(v0_avg, 1e-20), digits = 2)
                tab[i, 4] = format_value(v)
                tab[i, 5] = "$perc%"
            end
        end
        # TODO: Do this properly instead of via Jutul's import...
        Jutul.PrettyTables.pretty_table(io, tab, header=header, title = t)
    end

    pkeys = active_keys(dopt)
    print_table(pkeys, "Active optimization parameters")
    if print_inactive
        ikeys = inactive_keys(dopt)
        print_table(ikeys, "Inactive optimization parameters", false)
    end
end

function get_parameter_limits(x::DictParameters, key; throw = true)
    key = convert_key(key)
    val = get(x.parameter_targets, key, missing)
    if throw && ismissing(val)
        error("$key not found in limits")
    end
    return val
end

function get_nested_dict_value(x::AbstractDict, key)
    key = convert_key(key)
    for k in key
        x = x[k]
    end
    return x
end

function set_nested_dict_value!(x::AbstractDict, key, value)
    key = convert_key(key)
    for k in key[1:end-1]
        x = x[k]
    end
    if x[key[end]] isa Number
        value::Number
        x[key[end]] = value
    else
        x[key[end]] .= value
    end
    return x
end

function convert_key(x::KEYTYPE)
    return convert_key([x])
end

function convert_key(x::Vector)
    eltype(x)<:KEYTYPE
    return x
end


function widen_dict_copy(x::AbstractDict)
    new_dict = Jutul.OrderedDict()
    for (k, v) in pairs(x)
        new_dict[k] = widen_dict_copy(v)
    end
    return new_dict
end

function widen_dict_copy(x)
    return deepcopy(x)
end
