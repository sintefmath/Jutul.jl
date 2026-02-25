"""
    get_tstr(dT, lim = 3)

Get formatted time string of `dT` given in seconds, limited to `lim` number of units.
"""
function get_tstr(dT, lim = 3)
    if dT == 0
        return "start"
    elseif isinf(dT)
        return "steady state (∞)"
    else
        last_unit_s = TIME_UNITS_FOR_PRINTING[end][2]
        if dT < 0
            out = "-"
            dT = abs(dT)
        else
            out = ""
        end
        count = 1
        for (u, s) in TIME_UNITS_FOR_PRINTING
            is_last = count == lim || s == last_unit_s
            n = Int(floor(dT/u))
            if n > 0 || s == last_unit_s
                is_last = is_last || dT - n*u <= 0.0
                if is_last
                    n_str = @sprintf "%1.4g" dT/u
                    finalizer = "";
                else
                    n_str = "$n"
                    finalizer = ", "
                end
                dT -= n*u
                if n == 1
                    suffix = ""
                else
                    suffix = "s"
                end
                out *= "$n_str $(s)$suffix$finalizer"
                count += 1
                if is_last
                    break
                end
            end
        end
        return out
        # return Dates.canonicalize(Dates.CompoundPeriod(Millisecond(ceil(1000*dT))))
    end
end

function Base.show(io::IO, t::MIME"text/plain", sim::JutulSimulator)
    T = typeof(sim)
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

function overwrite_by_kwargs(cfg; throw = true, kwarg...)
    # Overwrite with varargin
    unused = Dict{Symbol, Any}()
    for (key, val) in pairs(kwarg)
        if throw || haskey(cfg, key)
            cfg[key] = val
        else
            unused[key] = val
        end
    end
    return unused
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
    fmt = raw"u. dd Y HH:MM"
    if length(sr.reports) == 0
        t = 0.0
    else
        if ismissing(sr.reports[1])
            t = 0.0
        else
            t = sum(x -> x[:total_time], sr.reports)
        end
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

function evaluate_objective(G, case::JutulCase, result::SimResult; kwarg...)
    packed_steps = AdjointPackedResult(result, case)
    model = case.model
    obj = adjoint_wrap_objective(G, model)
    return evaluate_objective(obj, model, packed_steps; input_data = case.input_data, kwarg...)
end
