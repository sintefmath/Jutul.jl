export add_option!

function add_option!(opts::JutulConfig, name::Symbol, default_value, short_description = "", value = missing; description = missing, valid_types = Any, valid_values = missing, replace = false)
    # Set up option and make sure default is actually valid, otherwise this will throw
    option = JutulOption(default_value, short_description, description, valid_types, valid_values)
    @assert !haskey(opts.options, name) || replace "Option :$name is already defined: $(opts.options[name]), cannot replace"
    opts.options[name] = option
    opts.values[name] = checked_value(default_value, valid_types, valid_values, name)
    # Might not be default
    if !ismissing(value)
        opts[name] = value
    end
    return opts
end

function checked_value(x, types, values, name)
    x = convert(types, x)
    if !ismissing(values)
        @assert x in values "$name must be one of $values, was: $x"
    end
    return x
end

function Base.getindex(opts::JutulConfig, name::Symbol)
    return opts.values[name]
end

function Base.setindex!(opts::JutulConfig, x, name::Symbol)
    if !haskey(opts.options, name)
        error("Option $name not found. It must be set up using `add_option!`.")
    end
    types = opts.options[name].valid_types
    values = opts.options[name].valid_values
    x = checked_value(x, types, values, name)
    opts.values[name] = x
    return opts
end

function Base.show(io::IO, t::MIME"text/plain", options::JutulConfig)
    header = ["Option", "Value", "Default", "Description", "Types", "Values"]
    vals = options.values
    opts = options.options
    vkeys = keys(options)
    n = length(vkeys)
    m = length(header)
    out = Matrix{Any}(undef, n, m)
    for (i, k) in enumerate(vkeys)
        opt = opts[k]
        v = vals[k]
        # Name in header
        out[i, 1] = k
        out[i, 2] = v
        out[i, 3] = v === opt.default_value
        descr = opt.short_description
        long_descr = opt.long_description
        if !ismissing(long_descr)
            descr = "$descr\n$long_descr"
        end
        if descr === ""
            descr = "<missing>"
        end
        out[i, 4] = descr
        out[i, 5] = opt.valid_types
        possible_vals = opt.valid_values
        if ismissing(possible_vals)
            possible_vals = "Any"
        end
        out[i, 6] = possible_vals
    end
    cw = zeros(Int, m)
    cw[4] = 60
    pretty_table(io,
        out,
        title = "$(options.name)",
        linebreaks = true,
        header = header,
        autowrap = true,
        alignment = :l,
        body_hlines = collect(1:n),
        columns_width = cw
        )
end

Base.iterate(opts::JutulConfig, arg...) = Base.iterate(opts.values, arg...)
Base.pairs(opts::JutulConfig) = Base.pairs(opts.values)
Base.haskey(opts::JutulConfig, name::Symbol) = Base.haskey(opts.values, name)
Base.keys(opts::JutulConfig) = Base.keys(opts.values)
Base.length(opts::JutulConfig) = Base.length(opts.values)
Base.propertynames(S::JutulConfig) = keys(S)
