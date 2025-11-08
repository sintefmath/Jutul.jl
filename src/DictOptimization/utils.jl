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
    if haskey(dopt.multipliers, parameter_name)
        mult = dopt.multipliers[parameter_name]
        vals = mult.value
        if is_max
            l = mult.abs_max
        else
            l = mult.abs_min
        end
        if !ismissing(mult.lumping) && l isa AbstractArray
            ix = get_lumping_first_entry(mult.lumping)
            l = l[ix]
        end
    else
        vals = get_nested_dict_value(dopt.parameters, parameter_name)
        lims = get_parameter_limits(dopt, parameter_name)
        l = realize_limit(vals, lims, is_max = is_max, strict = dopt.strict)
    end
    return l
end

function realize_limit(initial::Union{Number, Array}, lims::KeyLimits; is_max::Bool, strict::Bool = true)
    if is_max
        l = realize_limit_inner(initial, lims.rel_max, lims.abs_max, is_max = true, strict = strict)
    else
        l = realize_limit_inner(initial, lims.rel_min, lims.abs_min, is_max = false, strict = strict)
    end
    if !ismissing(lims.lumping) && l isa AbstractArray
        ix = get_lumping_first_entry(lims.lumping)
        l = l[ix]
    end
    return l
end

function limit_getindex(x::Number, I)
    return x
end

function limit_getindex(x::Array, I)
    return x[I]
end

function realize_limit_inner(initial::Array, rel, abs; is_max::Bool, strict = true)
    out = similar(initial)
    for (i, v) in enumerate(initial)
        r = limit_getindex(rel, i)
        a = limit_getindex(abs, i)
        out[i] = realize_limit_inner(v, r, a, is_max = is_max, strict = strict)
    end
    return out
end

function realize_limit_inner(initial::Number, rel_lim::Number, abs_lim::Number; is_max::Bool, strict::Bool = true)
    rel_delta = abs(initial*(rel_lim-1.0))
    if is_max
        l = min(abs_lim, initial + rel_delta)
        if strict
            @assert initial <= l
        end
    else
        l = max(abs_lim, initial - rel_delta)
        if strict
            @assert initial >= l
        end
    end
    return l
end

function realize_limits(dopt::DictParameters, parameter_name)
    lb = realize_limit(dopt, parameter_name, is_max = false)
    ub = realize_limit(dopt, parameter_name, is_max = true)
    return (min = lb, max = ub)
end

function realize_limits(dopt::DictParameters, x_setup::NamedTuple)
    lb = Float64[]
    ub = Float64[]
    for parameter_name in x_setup.names
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

function get_lumping_for_vectorize_nested(dopt::DictParameters)
    lumping = Dict()
    for k in active_keys(dopt)
        lims = get_parameter_limits(dopt, k, throw = false)
        if !ismissing(lims) && !ismissing(lims.lumping)
            firstix = get_lumping_first_entry(lims.lumping)
            lumping[k] = (first_index = firstix, lumping = lims.lumping)
        end
    end
    return lumping
end

function get_scaler_for_vectorize_nested(dopt::DictParameters)
    scalers = Dict()
    for k in active_keys(dopt)
        lims = get_parameter_limits(dopt, k, throw = false)
        if !ismissing(lims) && !ismissing(lims.scaler)
            scalers[k] = lims.scaler
        end
    end
    return scalers
end

function print_optimization_overview(dopt::DictParameters; io = Base.stdout, print_inactive = false)
    function fmt(x::Number)
        return "$(round(x, sigdigits=3))"
    end

    function fmt_lim(x; is_max)
        u = unique(x)
        if length(u) == 1 || length(x) > 3
            if is_max
                x = maximum(x)
            else
                x = minimum(x)
            end
            if isfinite(x)
                s = fmt(x)
            else
                s = "-"
            end
        else
            s = join(map(fmt, x), ", ")
        end
        return s
    end

    function avg(x)
        return x
    end

    function avg(x::AbstractArray)
        return sum(x)/length(x)
    end

    function format_value(x)
        return fmt(x)
    end

    function format_value(x::AbstractArray)
        u = unique(x)
        if length(x) > 3 || length(u) == 1
            a = avg(x)
            minval, maxval = extrema(x)
            maxdiff = max(abs(a - minval), abs(maxval - a))
            str = "$(round(a, sigdigits=3)) ± $(round(maxdiff, sigdigits=3))"
        else
            str = join(map(fmt, x), ", ")
        end
        return str
    end
    prm_opt = dopt.parameters_optimized
    is_optimized = !ismissing(prm_opt)

    function print_table(subkeys, t, print_opt = true)
        pt = dopt.parameter_targets
        prm = dopt.parameters
        header = ["Name", "Initial value", "Count", "Min", "Max"]
        alignment = [:r, :l, :r, :r, :r]
        if is_optimized && print_opt
            push!(header, "Optimized value")
            push!(header, "Change")
            push!(alignment, :l, :r)
        end
        tab = Matrix{Any}(undef, length(subkeys), length(header))
        for (i, k) in enumerate(subkeys)
            v0 = get_parameter_value(dopt, k, optimized = false)
            v0_avg = avg(v0)
            if haskey(pt, k)
                lims = realize_limits(dopt, k)
                limstr_min = fmt_lim(lims.min, is_max = false)
                limstr_max = fmt_lim(lims.max, is_max = true)
            else
                limstr_min = limstr_max = fmt_lim(NaN, is_max = false)
            end
            tab[i, 1] = join(k, ".")
            tab[i, 2] = format_value(v0)
            tab[i, 3] = length(v0)
            tab[i, 4] = limstr_min
            tab[i, 5] = limstr_max
            if is_optimized  && print_opt
                v = get_parameter_value(dopt, k, optimized = true)
                v_avg = avg(v)
                perc = round(100*(v_avg-v0_avg)/max(v0_avg, 1e-20), sigdigits = 2)
                tab[i, 6] = format_value(v)
                tab[i, 7] = "$perc%"
            end
        end
        PrettyTables.pretty_table(io, tab, header=header, title = t, alignment = alignment)
    end

    pkeys = active_keys(dopt)
    if length(pkeys) == 0
        println(io, "No active optimization parameters.")
    else
        print_table(pkeys, "Active optimization parameters")
    end
    if print_inactive
        ikeys = inactive_keys(dopt)
        if length(ikeys) == 0
            println(io, "No inactive optimization parameters.")
        else
            print_table(ikeys, "Inactive optimization parameters", false)
        end
    end
    multkeys = keys(dopt.multipliers)
    nmult = length(multkeys)
    if nmult == 0
        println(io, "No multipliers set.")
    else
        header = ["Name", "Targets", "Initial value", "Count", "Min", "Max"]
        alignment = [:r, :r, :l, :r, :r, :r]
        if is_optimized
            push!(header, "Optimized value")
            push!(header, "Change")
            push!(alignment, :l, :r)
        end
        tab = Matrix{Any}(undef, nmult, length(header))
        for (i, k) in enumerate(multkeys)
            mult = dopt.multipliers[k]
            mval = mult.value
            tab[i, 1] = k
            tab[i, 2] = join(map(t -> join(t, "."), mult.targets), "; ")
            tab[i, 3] = format_value(mval)
            tab[i, 4] = length(mval)
            tab[i, 5] = fmt_lim(mult.abs_min, is_max = false)
            tab[i, 6] = fmt_lim(mult.abs_max, is_max = true)
            if is_optimized
                tab[i, 7] = format_value(mult.optimized_value)
                perc = round(100*(avg(mult.optimized_value)-avg(mval))/max(avg(mval), 1e-20), sigdigits = 2)
                tab[i, 8] = "$perc%"
            end
        end
        pretty_table(io, tab, header=header, title = "Optimization multipliers", alignment = alignment)
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

function get_parameter_value(x::DictParameters, key; optimized::Bool = false)
    if optimized
        prm = x.parameters_optimized
    else
        prm = x.parameters
    end
    val = get_nested_dict_value(prm, key)
    L = get_parameter_limits(x, key, throw = false)
    if !ismissing(L) && !ismissing(L.lumping)
        ix = get_lumping_first_entry(L.lumping)
        val = val[ix]
    end
    return val
end

function get_lumping_first_entry(lumping::AbstractVector)
    vals = Int[]
    for i in 1:maximum(lumping)
        push!(vals, findfirst(isequal(i), lumping))
    end
    return vals
end

function get_lumping_first_entry(lumping)
    return get_lumping_first_entry(vec(lumping))
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
