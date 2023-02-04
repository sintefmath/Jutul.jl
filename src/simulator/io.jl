initialize_io(::Nothing) = nothing

function initialize_io(path)
    @assert isdir(path) "$path must be a valid directory for output."
end

function retrieve_output!(states, config, n)
    pth = config[:output_path]
    if config[:output_states] && !isnothing(pth)
        @debug "Reading states from $pth..."
        @assert isempty(states)
        read_results(pth, read_reports = true, states = states, verbose = config[:info_level] >= 0, range = 1:n);
    end
end

get_output_state(sim) = get_output_state(sim.storage, sim.model)


function store_output!(states, reports, step, sim, config, report)
    mem_out = config[:output_states]
    path = config[:output_path]
    file_out = !isnothing(path)
    # We always keep reports in memory since they are used for timestepping logic
    push!(reports, report)
    t_out = @elapsed if mem_out || file_out
        @tic "output state" state = get_output_state(sim)
        @tic "write" if file_out
            step_path = joinpath(path, "jutul_$step.jld2")
            @debug "Writing to $step_path"
            jldopen(step_path, "w") do file
                file["state"] = state
                file["report"] = report
                file["step"] = step
            end
        elseif mem_out
            push!(states, state)
        end
    end
    report[:output_time] = t_out
end
