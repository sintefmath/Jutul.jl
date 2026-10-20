"""
    data = convergence_monitor_data(reports; distance_function_args = NamedTuple())

Extract the data needed to replay the convergence monitor cutting criterion
offline for all ministeps (successful and failed) in the timestep `reports` of
a simulation. The reports can be given as a vector of timestep reports or a
simulation result with a `reports` property, or, for an ensemble of
simulations, as a vector of these. For each ministep, the function returns a
named tuple with fields

- `member`, `step`, `ministep`: ensemble member (one for a single simulation),
  timestep and ministep number.
- the fields returned by `ministep_convergence_data`.
"""
function convergence_monitor_data(reports; distance_function_args = NamedTuple())

    reports = report_list(reports)
    ensemble = !isempty(reports) && !(first(reports) isa AbstractDict)
    members = ensemble ? map(report_list, reports) : [reports]
    data = []
    for (member, member_reports) in enumerate(members)
        for (step_no, report) in enumerate(member_reports)
            haskey(report, :ministeps) || continue
            for (ministep_no, ministep) in enumerate(report[:ministeps])
                d = ministep_convergence_data(ministep; distance_function_args = distance_function_args)
                isnothing(d) && continue
                push!(data, (member = member, step = step_no, ministep = ministep_no, d...))
            end
        end
    end

    return data

end

report_list(reports::AbstractVector) = reports
report_list(result) = result.reports

"""
    d = ministep_convergence_data(ministep; distance_function_args = NamedTuple())

Extract the data needed to replay the convergence monitor cutting criterion
offline for a ministep report. Returns `nothing` if the ministep has no
iterations with residuals, and otherwise a named tuple with fields

- `outcome`: `:converged` for successful ministeps, `:failed` for ministeps that
  failed after reaching the maximum number of iterations or due to an exception,
  and `:censored` for ministeps that were stopped early (e.g., by a cutting
  criterion), for which it is unknown if they would have converged.
- `distance`: distance from convergence for each iteration (see
  `compute_distance`, to which `distance_function_args` are passed).
- `names`: names of the measures (columns of `distance`).
- `checked`: whether the cutting criterion is evaluated at each iteration (i.e.,
  the iteration did not converge and did not throw an exception).
- `solves`: cumulative number of linear solves after each iteration.
"""
function ministep_convergence_data(ministep; distance_function_args = NamedTuple())

    haskey(ministep, :steps) || return nothing
    steps = ministep[:steps]
    n = something(findfirst(s -> !haskey(s, :errors), steps), length(steps) + 1) - 1
    n > 0 || return nothing
    dist = [compute_distance(s[:errors]; distance_function_args...) for s in steps[1:n]]
    distance = reduce(vcat, [first(d)' for d in dist])
    names = last(first(dist))
    checked = [!get(s, :converged, false) && !haskey(s, :failure_exception) for s in steps[1:n]]
    solves = cumsum([Int(get(s, :solved, false)) for s in steps])
    # The last iteration of a ministep that reached the maximum number of
    # iterations is not solved, so a failed ministep with a solved last
    # iteration and no exception must have been stopped early
    final = steps[end]
    if ministep[:success]
        outcome = :converged
    elseif get(final, :solved, false) && !haskey(final, :failure_exception)
        outcome = :censored
    else
        outcome = :failed
    end

    return (
        outcome = outcome, distance = distance, names = names,
        checked = checked, solves = solves,
    )

end

"""
    nv = violation_count(distance; num_violations_cut = 3, strategy = :per_measure, kwargs...)

Compute the number of violations after each iteration of a distance history
(one row per iteration, one column per measure) that is compared to
`num_violations_cut` by the `ConvergenceMonitorCuttingCriterion`, i.e., the
maximum over the measures, or the sum for `strategy = :total`. The remaining
keyword arguments are passed to `analyze_convergence`.
"""
function violation_count(distance; num_violations_cut = 3, strategy = :per_measure, kwargs...)

    Nv = analyze_convergence(distance; strategy = strategy, kwargs...)[:num_violations]
    if strategy == :total
        return vec(sum(Nv, dims = 2))
    else
        return vec(maximum(Nv, dims = 2))
    end

end

"""
    k = replay_cutting_criterion(distance, checked; num_violations_cut = 3, kwargs...)

Replay the `ConvergenceMonitorCuttingCriterion` on a distance history (one row
per iteration, one column per measure) and return the iteration at which the
ministep would be cut, or `nothing` if it would not be cut. The cutting
criterion is only evaluated at iterations where `checked` is true. The
remaining keyword arguments are passed to `violation_count`.
"""
function replay_cutting_criterion(distance, checked; num_violations_cut = 3, kwargs...)

    nv = violation_count(distance; kwargs...)
    return findfirst(k -> checked[k] && nv[k] > num_violations_cut, eachindex(nv))

end

"""
    ev = evaluate_convergence_monitor(data; censored = :failed, kwargs...)

Evaluate the cutting criterion with parameters `kwargs` (see
`replay_cutting_criterion`) on the ministep `data` from
`convergence_monitor_data`. Returns a named tuple with

- `wasted_iterations`: number of linear solves spent on ministeps that do not
  converge, either because they fail or because they are cut.
- `false_rejections`: number of successful ministeps that would be cut.
- `true_rejections`: number of failed ministeps that would be cut.
- `missed_rejections`: number of failed ministeps that would not be cut.

Censored ministeps are treated as failed if `censored = :failed`, and ignored
if `censored = :exclude`. Since the iterations after the early stop are
unknown, the wasted iterations of a censored ministep that would not be cut is
a lower bound.
"""
function evaluate_convergence_monitor(data; censored = :failed, kwargs...)

    censored in (:failed, :exclude) || error("Unknown option censored = $censored")
    wasted, false_rej, true_rej, missed = 0, 0, 0, 0
    for d in data
        d.outcome == :censored && censored == :exclude && continue
        k = replay_cutting_criterion(d.distance, d.checked; kwargs...)
        if d.outcome == :converged
            if !isnothing(k)
                false_rej += 1
                wasted += d.solves[k]
            end
        elseif isnothing(k)
            missed += 1
            wasted += d.solves[end]
        else
            true_rej += 1
            wasted += d.solves[k]
        end
    end

    return (
        wasted_iterations = wasted, false_rejections = false_rej,
        true_rejections = true_rej, missed_rejections = missed,
    )

end

"""
    opt = optimize_convergence_monitor(reports, parameters::NamedTuple;
        distance_function_args = NamedTuple(),
        false_rejection_cost = nothing,
        censored = :failed,
        kwargs...
    )

Find the convergence monitor parameters that minimize the number of wasted
iterations and false rejections for the ministeps in the timestep `reports` of
a simulation, or of an ensemble of simulations (see `convergence_monitor_data`
for accepted formats), in which case the statistics are summed over all
ensemble members. The cutting criterion is replayed
offline on the residuals stored in the reports for all combinations of the
parameter values in `parameters`, e.g.,

    parameters = (theta_slow = [0.8, 0.9, 0.95], num_violations_cut = 1:5)

Valid parameters are `num_violations_cut` and the keyword arguments of
`analyze_convergence` (`memory`, `target_its`, `iteration_buffer`,
`iteration_margin`, `max_iterations_left`, `theta_fast`, `theta_slow`,
`strategy`). Parameters that are not varied can be fixed with `kwargs`, and
`target_its` should match the target of the `IterationTimestepSelector` used
in the simulation. The distance from convergence is computed with
`distance_function_args` (see `compute_distance`).

The parameter combinations are ranked by the objective

    wasted_iterations + false_rejection_cost * false_rejections

where `false_rejection_cost` defaults to the mean number of iterations of the
successful ministeps, i.e., the cost of solving a falsely rejected ministep
again. See `evaluate_convergence_monitor` for the definition of wasted
iterations and false rejections, and the handling of `censored` ministeps.

The analysis assumes that the iterates do not depend on the convergence monitor
parameters. The reports should therefore preferably be generated without
convergence monitor relaxation, and without a cutting criterion (or with a
large `num_violations_cut`) so that no ministeps are censored.

Returns a named tuple with fields

- `best`: the parameter combination with the lowest objective.
- `results`: all parameter combinations sorted by objective.
- `pareto`: the Pareto front in wasted iterations and false rejections, sorted
  by false rejections, with the best-ranked parameter combination for each
  point (other combinations attaining the same point can be found in
  `results`).
- `reference`: the evaluation without any cuts.
- `false_rejection_cost`: the cost used in the objective.
- `data`: the ministep data from `convergence_monitor_data`.
"""
function optimize_convergence_monitor(
        reports, parameters::NamedTuple;
        distance_function_args = NamedTuple(),
        false_rejection_cost = nothing,
        censored = :failed,
        kwargs...
    )

    data = convergence_monitor_data(reports; distance_function_args = distance_function_args)
    if isnothing(false_rejection_cost)
        conv = [d.solves[end] for d in data if d.outcome == :converged]
        false_rejection_cost = isempty(conv) ? 0.0 : sum(conv) / length(conv)
    end
    fixed = values(kwargs)
    results = []
    for vals in Iterators.product(values(parameters)...)
        prm = merge(fixed, NamedTuple{keys(parameters)}(vals))
        ev = evaluate_convergence_monitor(data; censored = censored, prm...)
        objective = ev.wasted_iterations + false_rejection_cost * ev.false_rejections
        push!(results, (parameters = prm, ev..., objective = objective))
    end
    sort!(results, by = r -> (r.objective, r.false_rejections))
    # Pareto front in (wasted iterations, false rejections)
    dominated(r) = any(
        o -> o.wasted_iterations <= r.wasted_iterations &&
            o.false_rejections <= r.false_rejections &&
            (o.wasted_iterations, o.false_rejections) != (r.wasted_iterations, r.false_rejections),
        results
    )
    pareto = unique(r -> (r.wasted_iterations, r.false_rejections), filter(!dominated, results))
    sort!(pareto, by = r -> r.false_rejections)
    reference = evaluate_convergence_monitor(
        data; censored = censored, num_violations_cut = typemax(Int), fixed...
    )

    return (
        best = first(results), results = results, pareto = pareto,
        reference = reference, false_rejection_cost = false_rejection_cost, data = data,
    )

end
