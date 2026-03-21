
function write_debug_output(config, storage, model, variant::Symbol)
    if variant == :equations
        is_equations = true
    elseif variant == :state
        is_equations = false
    else
        error("Unknown debug output variant: $variant")
    end
    rec = progress_recorder(storage)
    # Set offset - we are writing before the iteration is incremented
    pth = debug_output_path(config, rec, variant, it_offset = 1)
    if !ismissing(pth)
        @debug "Writing debug state to $pth"
        jldopen(pth, "w") do file
            if is_equations
                file["equations"] = storage.views
            else
                file["state"] = get_output_state(storage, model)
            end
            file["iteration"] = iteration(rec) + 1
            file["subiteration"] = subiteration(rec) + 1
            file["step"] = step(rec)
            file["substep"] = substep(rec)
            file["time"] = recorder_current_time(rec, :global)
        end
    end
end

function debug_output_path(config::JutulConfig, rec, variant; it_offset = 0)
    if variant == :equations
        basepth = config[:debug_equations_path]
    elseif variant == :state
        basepth = config[:debug_states_path]
    else
        error("Unknown debug output variant: $variant")
    end
    if ismissing(basepth)
        out = missing
    else
        mkpath(basepth)
        out = debug_output_path(basepth, rec, variant, it_offset = it_offset)
    end
    return out
end

function debug_output_path(basepth::String, rec, variant; it_offset = 0)
    s = step(rec)
    ss = substep(rec)
    it = iteration(rec) + it_offset
    subit = subiteration(rec) + it_offset
    out = debug_output_path(basepth, variant, s, ss, it, subit)
    return out
end

function debug_output_path(basepth, variant, step, substep, it, subit)
    out = joinpath(basepth, "$(variant)_jutul_newton_step_$(step)_substep_$(substep)_$(it)_sub_$(subit).jld2")
    return out
end

function read_debug_output(path::String, variant::Symbol)
    isdir(path) || error("Path $path does not exist or is not a directory")
    out = []
    done = false
    for step in 1:typemax(Int)
        for substep in 1:typemax(Int)
            m = length(out)
            for it in 1:typemax(Int)
                n = length(out)
                read_debug_output(path, variant, step, substep, it)
                if length(out) == n
                    break
                end
            end
            done = length(out) == m
            if done
                break
            end
        end
        if done
            break
        end
    end
    return out
end

function read_debug_output(path, variant; step::Int = 1, substep::Int = 1, it::Int = 1)
    # Utility to read all Newton iterations for a given outer iteration
    out = []
    return read_debug_output!(out, path, variant, step, substep, it)
end

function read_debug_output!(out, path, variant, step::Int, substep::Int, it::Int)
    for subit in 1:typemax(Int)
        next = read_debug_output(path, variant, step, substep, it, subit)
        if ismissing(next)
            break
        end
        push!(out, next)
    end
    return out
end

function read_debug_output(path, variant, step::Int, substep::Int, it::Int, subit::Int)
    fpth = debug_output_path(path, variant, step, substep, it, subit)
    if isfile(fpth)
        @debug "Reading debug output from $fpth"
        data = jldopen(fpth, "r") do file
            Dict(Symbol(k) => file[k] for k in keys(file))
        end
        out = data
    else
        out = missing
    end
    return out
end
