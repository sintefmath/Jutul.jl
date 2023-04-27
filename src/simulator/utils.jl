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
        cfg[key] = kwarg[key]
    end
end

function Base.iterate(t::SimResult)
    return (t.states, :states)
end

function Base.iterate(t::SimResult, state)
    @assert state == :states
    return (t.reports, nothing)
end

function Base.getindex(t::SimResult, i::Int)
    if length(t.states) < i
        state = nothing
    else
        state = t.states[i]
    end
    return (state = state, report = t.reports[i])
end

function Base.show(io::IO, ::MIME"text/plain", sr::SimResult)
    function print_keys(prefix, el)
        for k in keys(el)
            v = el[k]
            if v isa AbstractDict
                print(io, "$prefix:$k\n")
                print_keys("  $prefix", v)
            else
                if v isa AbstractVecOrMat
                    s = " of size $(size(v))"
                else
                    s = ""
                end
                print(io, "$prefix:$k => $(typeof(v))$s\n")
            end
        end
    end
    states = sr.states
    n = length(states)
    print(io, sr)
    print(io, ":\n\n")
    if n > 1
        el = first(states)
        print(io, "  states (model variables)\n")
        print_keys("    ", el)
        print(io, "\n  reports (timing/debug information)\n")
        print_keys("    ", first(sr.reports))
    end
    print_sim_result_timing(io, sr)
end

function print_sim_result_timing(io, sr::SimResult)
    fmt = raw"u. dd Y H:mm"
    if length(sr.reports) == 0
        t = 0.0
    else
        t = sum(x -> x[:total_time], sr.reports)
    end
    print(io, "\n  Completed at $(Dates.format(sr.start_timestamp, fmt)) after $(get_tstr(t)).")
end

function Base.show(io::IO, sr::SimResult)
    n = length(sr.states)
    if n == 1
        s = "entry"
    else
        s = "entries"
    end
    print(io, "SimResult with $n $s")
end

