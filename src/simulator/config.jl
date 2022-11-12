function simulator_config!(cfg, sim; kwarg...)
    # Maximum number of cuts before a time-step is terminated
    cfg[:max_timestep_cuts] = 5
    # Maximum number of nonlinear iterations before a time-step is cut
    cfg[:max_nonlinear_iterations] = 15
    # Minimum number of nonlinear solves performed, even when equations are converged
    cfg[:min_nonlinear_iterations] = 1
    # Always update secondary variables - only useful for nested solvers
    cfg[:always_update_secondary] = false
    # Throw an error if the solve
    cfg[:error_on_incomplete] = false
    # The linear solver used to solve linearized systems
    cfg[:linear_solver] = nothing
    # Path for output
    cfg[:output_path] = nothing
    # Produce states for output (keeping them in memory)
    cfg[:output_states] = nothing
    # Extra checks on values etc
    cfg[:safe_mode] = true
    # Define debug level. If debugging is on, this determines the amount of output.
    cfg[:debug_level] = 1
    # Info level determines the runtime output to the terminal:
    # < 0 - no output.
    # 0   - gives minimal output (just a progress bar by default, and a final report)
    # 1   - gives some more details, printing at the start of each step
    # 2   - as 1, but also printing the current worst residual at each iteration
    # 3   - as 1, but prints a table of all non-converged residuals at each iteration
    # 4   - as 3, but all residuals are printed (even converged values)
    # The interpretation of this number is subject to change
    cfg[:info_level] = 0
    # Output extra, highly detailed performance report at simulation end
    cfg[:extra_timing] = false
    # Avoid unicode (if possible)
    cfg[:ascii_terminal] = false
    # Define a default progress ProgressRecorder
    cfg[:ProgressRecorder] = ProgressRecorder()
    cfg[:timestep_selectors] = [TimestepSelector()]
    cfg[:timestep_max_increase] = 10.0
    cfg[:timestep_max_decrease] = 0.1
    # Max residual before error is issued
    cfg[:max_residual] = 1e20
    cfg[:end_report] = nothing
    # Tolerances
    cfg[:tolerances] = set_default_tolerances(sim.model)

    overwrite_by_kwargs(cfg; kwarg...)
    if isnothing(cfg[:end_report])
        cfg[:end_report] = cfg[:info_level] > -1
    end
    # Default: Do not store states in memory if output to file.
    if isnothing(cfg[:output_states])
        cfg[:output_states] = isnothing(cfg[:output_path])
    end
    # Ensure that the folder exists, if requested.
    pth = cfg[:output_path]
    if !isnothing(pth) && !isdir(pth)
        @assert isa(pth, String)
        @debug "Creating $pth for output."
        mkpath(pth)
    end
    if cfg[:ascii_terminal]
        fmt = tf_markdown
    else
        fmt = tf_unicode_rounded
    end
    cfg[:table_formatter] = fmt;
    return cfg
end

function simulator_config(sim; kwarg...)
    cfg = Dict()
    simulator_config!(cfg, sim; kwarg...)
    return cfg
end
