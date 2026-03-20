
function write_debug_output(config, storage, model, variant::Symbol)
    if variant == :equations
        is_equations = true
    elseif variant == :state
        is_equations = false
    else
        error("Unknown debug output variant: $variant")
    end
    rec = progress_recorder(storage)
    pth = debug_output_path(config, rec, variant)
    if !ismissing(pth)
        @debug "Writing debug state to $pth"
        jldopen(pth, "w") do file
            if is_equations
                file["equations"] = storage.views
            else
                file["state"] = get_output_state(storage, model)
            end
            file["iteration"] = iteration(rec)
            file["subiteration"] = subiteration(rec)
            file["step"] = step(rec)
            file["substep"] = substep(rec)
            file["time"] = recorder_current_time(rec, :global)
        end
    end
end

function debug_output_path(config::JutulConfig, rec, variant)
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
        out = debug_output_path(basepth, rec, variant)
    end
    return out
end

function debug_output_path(basepth::String, rec, variant)
    s = step(rec)
    ss = substep(rec)
    it = iteration(rec)
    subit = subiteration(rec)
    out = joinpath(basepth, "$(variant)_jutul_newton_$(it)_sub_$(subit)_step_$(s)_substep_$(ss).jld2")
    return out
end
