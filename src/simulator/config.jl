function simulator_config!(cfg, sim; output_unused = false, nonlinear_tolerance = 1e-3, kwarg...)
    # Printing, etc
    add_option!(cfg, :info_level, 0, "Info level determines the amount of runtime output to the terminal during simulation.", types = Union{Int, Float64},
    description = "
0 - gives minimal output (just a progress bar by default, and a final report)
1 - gives some more details, printing at the start of each step
2 - as 1, but also printing the current worst residual at each iteration
3 - as 1, but prints a table of all non-converged residuals at each iteration
4 - as 3, but all residuals are printed (even converged values)
Negative values disable output. The interpretation of this number is subject to change.")
    add_option!(cfg, :debug_level, 0, "Define the amount of debug output in the reports. Higher values means more output.", types = Int)
    add_option!(cfg, :end_report, nothing, "Output a final report that includes timings etc. If nothing, depends on info_level instead.", types = Union{Bool, Nothing})
    add_option!(cfg, :id, "", "String identifier for simulator that is prefixed to some verbose output.", types = String)

    # Convergence tests
    add_option!(cfg, :max_timestep_cuts, 5, "Max time step cuts in a single mini step before termination of simulation.", types = Int, values = 0:10000)
    add_option!(cfg, :max_timestep, Inf, "Max time step length.", types = Float64)
    add_option!(cfg, :min_timestep, 0.0, "Min time step length.", types = Float64)
    add_option!(cfg, :termination_criterion, NoTerminationCriterion(), "Termination criterion (type instance). See AbstractTerminationCriterion for more details.", types = AbstractTerminationCriterion)
    add_option!(cfg, :max_nonlinear_iterations, 15, "Max number of nonlinear iterations in a Newton solve before time-step is cut.", types = Int, values = 0:10000)
    add_option!(cfg, :min_nonlinear_iterations, 1, "Minimum number of nonlinear iterations in Newton solver.", description = "This number of Newtion iterations is always performed, even if all equations are converged.", types = Int, values = 0:10000)
    add_option!(cfg, :failure_cuts_timestep, false, "Cut the timestep if exceptions occur during step. If set to false, throw errors and terminate.", types = Bool)
    add_option!(cfg, :check_before_solve, true, "Check convergence before solving linear system. Can skip some linear solves if not using increment tolerances.", types = Bool)

    # Boolean options
    add_option!(cfg, :always_update_secondary, false, "Always update secondary variables (even when they can be reused from end of previous step). Only useful for nested solvers", types = Bool)
    add_option!(cfg, :error_on_incomplete, false, "Throw an error if the simulation could not complete. If `false` emit a message and return.", types = Bool)
    add_option!(cfg, :output_states, true, "Return states in-memory as output.",
    description = "For larger models with many time-steps, using `output_path` might be better to avoid filling up your memory.", types = Bool)
    add_option!(cfg, :output_reports, true, "Return reports in-memory as output.", types = Bool)
    add_option!(cfg, :safe_mode, true, "Add extra checks in simulator that have a small extra cost.", types = Bool)
    add_option!(cfg, :extra_timing, get(ENV, "JUTUL_EXTRA_TIMING", false), "Output extra, highly detailed performance report at simulation end.", 
    description = " This uses TimerOutputs.jl's @timeit_debug macro. You may have to call the function twice with this option the first time you use it.", types = Bool)
    add_option!(cfg, :ascii_terminal, false, "Avoid unicode (if possible) in terminal output.", types = Bool)

    # Linear, nonlinear solver
    add_option!(cfg, :linear_solver, select_linear_solver(sim), "The linear solver used to solve linearized systems.")
    add_option!(cfg, :timestep_selectors, [TimestepSelector()], "Time-step selectors that pick mini steps.")
    add_option!(cfg, :timestep_max_increase, 10.0, "Max allowable factor to increase time-step by. Overrides step selectors.", types = Float64)
    add_option!(cfg, :timestep_max_decrease, 0.1, "Max allowable factor to decrease time-step by. Overrides step selectors.", types = Float64)
    add_option!(cfg, :max_residual, 1e20, "Maximum value allowed for a residual before simulation is terminated.", types = Float64)
    add_option!(cfg, :relaxation, NoRelaxation(), "Non-Linear relaxation used. Currently supports `NoRelaxation()` and `SimpleRelaxation()`.", types = NonLinearRelaxation)
    add_option!(cfg, :cutting_criterion, nothing, "Criterion to use for early cutting of time-steps. Default value of nothing means cutting when max_nonlinear_iterations is reached.")

    # Tolerances
    add_option!(cfg, :tolerances, set_default_tolerances(sim, tol = nonlinear_tolerance), "Tolerances used for convergence criterions.")
    add_option!(cfg, :tol_factor_final_iteration, 1.0, "Value that multiplies all tolerances for the final convergence check before a time-step is cut.")

    # IO options
    add_option!(cfg, :output_path, nothing, "Path to write output. If nothing, output is not written to disk.", types = Union{String, Nothing})
    add_option!(cfg, :in_memory_reports, 10, "Limit for number of reports kept in memory if output_path is provided.", types = Int)
    add_option!(cfg, :report_level, 0, "Level of information stored in reports when written to disk.", types = Int)
    add_option!(cfg, :output_substates, false, "Store substates (between report steps) as field on each state.", types = Bool)
    add_option!(cfg, :output_function, missing, "Function on the form (state, report) -> state that is run before output is written to disk or returned. This can be used to remove fields or add data. Note that it is easy to break the restart functionality if you modify the state in a non-compatible way. Use with care.")

    # Hooks
    add_option!(cfg, :post_ministep_hook, missing, "Hook to run after each ministep (successful or not) on format (done, report, sim, dt, forces, max_iter, cfg) -> (done, report)")
    add_option!(cfg, :post_iteration_hook, missing, "Hook to run after each iteration on format (converged, report, storage, model, dt, forces, cfg, iteration) -> converged")

    add_option!(cfg, :prepare_step, missing, "Type instance that get called with prepare_step before each Newton iteration.")

    prepare_step_handler, prepare_step_storage = get_prepare_step_handler(sim)
    simulator_config!(cfg, sim, prepare_step_handler, prepare_step_storage)

    # Fine grained control over printing
    add_option!(cfg, :progress_color, :green, "Color for progress meter.", types = Symbol)
    add_option!(cfg, :progress_glyphs, :default, "Glyphs", types = Union{Symbol, ProgressMeter.BarGlyphs})

    unused = overwrite_by_kwargs(cfg; throw = !output_unused, kwarg...)
    if isnothing(cfg[:end_report])
        cfg[:end_report] = cfg[:info_level] > -1
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
    add_option!(cfg, :table_formatter, fmt, "Formatter for tables.")
    if output_unused
        out = (cfg, unused)
    else
        out = cfg
    end
    return out
end

function simulator_config!(cfg, sim, ::Missing, ::Missing)
    # Do nothing
end

"""
    simulator_config(sim; info_level = 3, linear_solver = GenericKrylov())

Set up a simulator configuration object that can be passed onto
[`simulate!`](@ref).

There are many options available to configure a given simulator. The best way to
get an overview of these possible configuration options is to instatiate the
config without any arguments and inspecting the resulting table by calling
`simulator_config(sim)` in the REPL.
"""
function simulator_config(sim; kwarg...)
    cfg = JutulConfig("Simulator config")
    simulator_config!(cfg, sim; kwarg...)
    return cfg
end

function select_linear_solver(sim::Simulator; kwarg...)
    return select_linear_solver(sim.model; kwarg...)
end

select_linear_solver(model::JutulModel; kwarg...) = nothing # LUSolver(; kwarg...)
