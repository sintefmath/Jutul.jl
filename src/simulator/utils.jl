function get_tstr(dT)
    if dT == 0
        return "start"
    else
        return Dates.canonicalize(Dates.CompoundPeriod(Millisecond(ceil(1000*dT))))
    end
end

function Base.show(io::IO, t::MIME"text/plain", sim::T) where T<:JutulSimulator
    println(io, "$T:")
    for f in fieldnames(typeof(sim))
        p = getfield(sim, f)
        print(io, "  $f:\n")
        if f == :storage
            for key in keys(sim.storage)
                ss = sim.storage[key]
                println(io, "    $key")
            end
        else
            print(io, "    ")
            show(io, t, p)
            print(io, "\n\n")
        end
    end
end

function overwrite_by_kwargs(cfg; kwarg...)
    # Overwrite with varargin
    for key in keys(kwarg)
        if !haskey(cfg, key)
            @warn "Key $key is not found in default config. Misspelled?"
        end
        cfg[key] = kwarg[key]
    end
end

