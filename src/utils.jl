export convert_to_immutable_storage, convert_to_mutable_storage, gravity_constant, report_stats, print_stats
export get_cell_faces, get_facepos, get_cell_neighbors

const gravity_constant = 9.80665

function convert_to_immutable_storage(dct::AbstractDict)
    for (key, value) in dct
        dct[key] = convert_to_immutable_storage(value)
    end
    return (; (Symbol(key) => value for (key, value) in dct)...)
end

function convert_to_immutable_storage(v::Any)
    # Silently do nothing
    return v
end

function convert_to_mutable_storage(v::NamedTuple)
    D = Dict{Symbol, Any}()
    for k in keys(v)
        D[k] = convert_to_mutable_storage(v[k])
    end
    return D
end

function convert_to_mutable_storage(v::Any)
    # Silently do nothing
    return v
end

function as_cell_major_matrix(v, n, m, model::SimulationModel, offset = 0)
    transp = !is_cell_major(matrix_layout(model.context))
    return get_matrix_view(v, n, m, transp, offset)
end

function get_matrix_view(v0, n, m, transp = false, offset = 0)
    if size(v0, 2) == 1
        r_l = view(v0, (offset+1):(offset + n*m))
        if transp
            v = reshape(r_l, m, n)'
        else
            v = reshape(r_l, n, m)
        end
    else
        v = view(v0, (offset+1):(offset+n), :)
        if transp
            v = v'
        end
    end
    return v
end

function check_increment(dx, pvar, key)
    has_bad_values = any(!isfinite, dx)
    if has_bad_values
        bad = findall(isfinite.(dx) .== false)
        n_bad = length(bad)
        n = min(10, length(bad))
        bad = bad[1:n]
        @warn "$key: $n_bad non-finite values found. Indices: (limited to 10) $bad"
    end
    ok = !has_bad_values
    return ok
end

# function get_row_view(v::AbstractVector, n, m, row, transp = false, offset = 0)
#     v = get_matrix_view(v, n, m, transp, offset)
#     view(v, row, :)
# end

function get_convergence_table(errors::AbstractDict, arg...)
    # Already a dict
    conv_table_fn(errors, true, arg...)
end

function get_convergence_table(errors, arg...)
    d = OrderedDict()
    d[:Base] = errors
    conv_table_fn(d, false, arg...)
end

function conv_table_fn(model_errors, has_models, info_level, iteration, cfg)
    # Info level 1: Function should never be called
    # Info level 2:
    # Info level 3: Just print the non-converged parts?
    # Info level 4+: Full print
    if info_level == 1
        return
    end
    id = haskey(cfg, :id) ? "$(cfg[:id]): " : ""
    fmt = cfg[:table_formatter]
    count_crit = 0
    count_ok = 0
    worst_val = 0.0
    worst_tol = 0.0
    worst_name = ""
    print_converged = info_level > 3
    header = ["Equation", "Type", "Name", "‖R‖", "ϵ"]
    alignment = [:l, :l, :l, :r, :r]
    if has_models
        # Make the code easier to have for both multimodel and single model case.
        header = ["Model", header...]
        alignment = [:l, alignment...]
    end
    tbl = []
    tols = Vector{Float64}()
    body_hlines = Vector{Int64}()
    pos = 1
    # Loop over models
    for (model, equations) in model_errors
        # Loop over equations 
        for (mix, eq) in enumerate(equations)
            criterions = eq.criterions
            tolerances = eq.tolerances
            eq_touched = false
            for crit in keys(criterions)
                C = criterions[crit]
                local_errors = C.errors
                local_names = C.names
                tol = tolerances[crit]
                touched = false
                for (i, e) in enumerate(local_errors)
                    touch = print_converged || e > tol
                    if touch && !touched
                        nm = eq.name
                        T = crit
                        tt = tol
                    else
                        nm = ""
                        T = ""
                        tt = ""
                    end
                    touched = touch || touched

                    if touch
                        nm2 = local_names[i]
                        if has_models
                            if eq_touched
                                m = ""
                            else
                                m = String(model)
                            end
                            t = [m nm T nm2 e tt]
                        else
                            t = [nm T nm2 e tt]
                        end
                        push!(tbl, t)
                        push!(tols, tol)
                        pos += 1
                        eq_touched = true
                    end
                    count_crit += 1
                    count_ok += e <= tol
                    e_scale = e/tol
                    if e_scale > worst_val
                        worst_val = e_scale
                        worst_tol = tol
                        if has_models
                            mstr = "from model $(UNDERLINE(String(model)))"
                        else
                            mstr = ""
                        end
                        pref = "$(eq.name) ($(local_names[i]))"
                        worst_name = "$(UNDERLINE("$pref")) $mstr"
                    end
                end
                push!(body_hlines, pos-1)
            end
        end
    end
    tbl = vcat(tbl...)
    print_table = size(tbl, 1) > 0 && info_level > 2
    max_its = cfg[:max_nonlinear_iterations]
    if print_table
        if info_level == 3
            s  = ". Non-converged:"
        else
            s  = ". All criteria:"
        end
    elseif count_crit == count_ok
        s = " ✔️"
    else
        worst_print = @sprintf "%2.3e (ϵ = %g)" worst_val*worst_tol worst_tol
        s = ". Worst value:\n\t - $worst_name at $worst_print."
    end
    # @info "$(id)It. $iteration/$max_its: $count_ok/$count_crit criteria converged$s"
    jutul_message("It. $(iteration-1)/$max_its", "$count_ok/$count_crit criteria converged$s", color = :cyan)
    if print_table
        m_offset = Int64(has_models)
        rpos = (4 + m_offset)
        nearly_factor = 10
        function not_converged(data, i, j)
            if j == rpos
                d = data[i, j]
                t = tols[i]
                return d > t && d > 10*t
            else
                return false
            end
        end
        h1 = Highlighter(f = not_converged, crayon = crayon"red" )
        function nearly_converged(data, i, j)
            if j == rpos
                d = data[i, j]
                t = tols[i]
                return d > t && d < nearly_factor*t
            else
                return false
            end
        end
        h2 = Highlighter(f = nearly_converged, crayon = crayon"yellow")
        function converged(data, i, j)
            if j == rpos
                return data[i, j] <= tols[i]
            else
                return false
            end
        end
        h3 = Highlighter(f = converged, crayon = crayon"green")
        highlighers = (h1, h2, h3)
        pretty_table(tbl, header = header,
                                alignment = alignment, 
                                body_hlines = body_hlines,
                                highlighters = highlighers,
                                tf = fmt,
                                formatters = ft_printf("%2.3e", [m_offset + 4]),
                                crop=:none)
    end
end

function initialize_report_stats(reports)
    stats = Dict{Symbol, Union{Int64, Float64}}(:wasted_iterations => 0,
                                                :iterations => 0,
                                                :steps => length(reports),
                                                :ministeps => 0,
                                                :wasted_linearizations => 0,
                                                :wasted_linear_iterations => 0,
                                                :linear_update => 0.0,
                                                :linear_solve => 0.0,
                                                :linear_setup => 0.0,
                                                :linear_iterations => 0,
                                                :linearizations => 0,
                                                :finalize => 0.0,
                                                :secondary => 0.0,
                                                :equations => 0.0,
                                                :update => 0.0,
                                                :convergence => 0.0,
                                                :io => 0.0,
                                                :time => 0.0,
    )
    return stats
end

function outer_step_report_stats!(stats, outer_rep)
    stats[:time] += outer_rep[:total_time]
    if haskey(outer_rep, :output_time)
        t_io = outer_rep[:output_time]
        stats[:io] += t_io
        stats[:time] += t_io
    end
    for mini_rep in outer_rep[:ministeps]
        ministep_report_stats!(stats, mini_rep)
    end
end

function ministep_report_stats!(stats, mini_rep)
    stats[:ministeps] += 1
    if haskey(mini_rep, :finalize_time)
        stats[:finalize] += mini_rep[:finalize_time]
    end

    s = stats_ministep(mini_rep[:steps])
    stats[:linearizations] += s.linearizations
    stats[:iterations] += s.newtons
    stats[:linear_update] += s.linear_system
    stats[:linear_solve] += s.linear_solve
    stats[:linear_setup] += s.linear_setup
    stats[:linear_iterations] += s.linear_iterations
    stats[:update] += s.update
    stats[:equations] += s.equations
    stats[:secondary] += s.secondary
    stats[:convergence] += s.convergence

    if !mini_rep[:success]
        stats[:wasted_iterations] += s.newtons
        stats[:wasted_linearizations] += s.linearizations
        stats[:wasted_linear_iterations] += s.linear_iterations
    end
end

function summarize_report_stats(stats, per = false)
    if per
        linscale = v -> v / max(stats[:linearizations], 1)
        itscale = v -> v / max(stats[:iterations], 1)
        miniscale = v -> v / max(stats[:ministeps], 1)
    else
        linscale = itscale = miniscale = identity
    end
    summary = (
                secondary = itscale(stats[:secondary]), # Not always updated, itscale
                equations = linscale(stats[:equations]),
                linear_system = linscale(stats[:linear_update]),
                linear_solve = itscale(stats[:linear_solve]),
                linear_setup = itscale(stats[:linear_setup]),
                update = itscale(stats[:update]),
                convergence = linscale(stats[:convergence]),
                io = miniscale(stats[:io]),
                other = itscale(stats[:other_time]),
                total = itscale(stats[:time])
            )
    return summary
end

function update_other_time_report_stats!(stats)
    sum_measured = 0.0
    for k in [:equations, :secondary, :linear_update, :linear_setup, :linear_solve, :update, :convergence, :io]
        sum_measured += stats[k]
    end
    stats[:other_time] = stats[:time] - sum_measured
    return stats
end

function output_report_stats(stats)
    totals = summarize_report_stats(stats, false)
    each = summarize_report_stats(stats, true)

    out = (
        newtons = stats[:iterations],
        linearizations = stats[:linearizations],
        linear_iterations = stats[:linear_iterations],
        wasted = (newtons = stats[:wasted_iterations],
                  linearizations = stats[:wasted_linearizations],
                  linear_iterations = stats[:wasted_linear_iterations]),
        steps = stats[:steps],
        ministeps = stats[:ministeps],
        time_sum = totals,
        time_each = each
       )
    return out
end

function report_stats(reports)
    stats = initialize_report_stats(reports)
    for outer_rep in reports
        outer_step_report_stats!(stats, outer_rep)
    end
    update_other_time_report_stats!(stats)
    return output_report_stats(stats)
end

function stats_ministep(reports)
    linearizations = 0
    its = 0
    secondary = 0
    equations = 0
    convergence = 0
    linear_system = 0
    update = 0
    linsolve = 0
    linprep = 0
    linear_iterations = 0
    for rep in reports
        linearizations += 1
        its += rep[:solved]
        if haskey(rep, :update_time)
            if !rep[:solved]
                @warn "Strange data. solved = false but contains :update_time?"
            end
            update += rep[:update_time]
            lprep = rep[:linear_solver].prepare
            linsolve += rep[:linear_solve_time] - lprep
            linprep += lprep
            linear_iterations += rep[:linear_iterations]
        end
        secondary += rep[:secondary_time]
        equations += rep[:equations_time]
        linear_system += rep[:linear_system_time]
        convergence += rep[:convergence_time]
    end
    return (linearizations = linearizations,
            newtons = its, 
            equations = equations,
            secondary = secondary,
            convergence = convergence,
            update = update,
            linear_iterations = linear_iterations,
            linear_solve = linsolve,
            linear_setup = linprep,
            linear_system = linear_system)
end

function report_iterations(reports::AbstractVector)
    return sum(report_iterations, reports)
end

function report_iterations(report)
    its = 0
    if haskey(report, :ministeps)
        for rep in report[:ministeps]
            if haskey(rep, :update_time)
                its += 1
            end
        end
    end
    return its
end


function pick_time_unit(t, wide = is_wide_term())
    m = maximum(t)
    if wide
        day = "Days"
        hours = "Hours"
        min = "Minutes"
        sec = "Seconds"
        millisec = "Milliseconds"
        microsec = "Microseconds"
        nanosec = "Nanoseconds"
        picosec = "Picoseconds"
    else
        day = "day"
        hours = "h"
        min = "min"
        sec = "s"
        millisec = "ms"
        microsec = "μs"
        nanosec = "ns"
        picosec = "ps"
    end
    units = [(24*3600, day),
             (3600, hours),
             # (60, min),
             (1, sec),
             (1e-3, millisec),
             (1e-6, microsec),
             (1e-9, nanosec),
             (1e-12, picosec)]
    for u in units
        if m > u[1]
            return u
        end
    end
    # Fallback
    return (1, sec)
end

function autoformat_time(t::Float64; compact = true)
    u, s = pick_time_unit(t, !compact)
    t_fmt = @sprintf("%.2f", t/u)
    return "$t_fmt $s"
end

function print_stats(reports::AbstractArray, io = stdout; kwarg...)
    stats = report_stats(reports)
    print_stats(stats, io; kwarg...)
end

function print_stats(stats, io = stdout; title = "", table_formatter = tf_unicode_rounded, kwarg...)
    print_iterations(stats, io; title = title, table_formatter = table_formatter, kwarg...)
    print_timing(stats, io; title = title, table_formatter = table_formatter)
end

function is_wide_term()
    _, dim = displaysize(stdout)
    return dim > 90
end

function print_iterations(stats, io = stdout;
        title = "",
        table_formatter = tf_unicode_rounded,
        scale = 1
    )
    flds = (:newtons, :linearizations, :linear_iterations)
    names = [:Newton, :Linearization, Symbol("Linear solver")]
    data = Array{Any}(undef, length(flds), 4)
    if scale == 1
        sf = identity
    else
        sf = x -> x/scale
    end
    nstep = sf(stats.steps)
    nmini = sf(stats.ministeps)

    tot_time = stats.time_sum.total
    time = map(f -> tot_time/stats[f], flds)
    u, s = pick_time_unit(time)

    for (i, f) in enumerate(flds)
        waste = sf(stats[:wasted][f])
        raw = sf(stats[f])
        data[i, 1] = raw/nstep         # Avg per step
        data[i, 2] = raw/nmini         # Avg per mini
        data[i, 3] = time[i]/u         # Time each
        data[i, 4] = "$raw ($waste)"    # Total
    end

    pretty_table(io, data; header = (["Avg/step", "Avg/ministep", "Time per", "Total"],
                                ["$nstep steps", "$nmini ministeps", s, "(wasted)"]), 
                      row_labels = names,
                      title = title,
                      title_alignment = :c,
                      row_label_alignment = :l,
                      tf = table_formatter,
                      formatters = (ft_printf("%3.4f", 3)),
                      row_label_column_title = "Iteration type")
end

function print_timing(stats, io = stdout; title = "", table_formatter = tf_unicode_rounded)
    flds = collect(keys(stats.time_each))
    n = length(flds)

    data = Array{Any}(undef, n, 3)
    tot = stats.time_sum.total
    for (i, f) in enumerate(flds)
        teach = stats.time_each[f]
        tsum = stats.time_sum[f]
        data[i, 1] = teach
        data[i, 2] = 100*tsum/tot
        data[i, 3] = tsum
    end

    u, s = pick_time_unit(data[:, 1])
    u_t, s_t = pick_time_unit(data[:, 3])

    @. data[:, 1] /= u
    @. data[:, 3] /= u_t

    function translate_for_table(name)
        if name == :equations
            name = :Equations
        elseif name == :secondary
            name = :Properties
        elseif name == :linear_system
            name = :Assembly
        elseif name == :linear_solve
            name = Symbol("Linear solve")
        elseif name == :linear_setup
            name = Symbol("Preconditioner")
        elseif name == :update
            name = :Update
        elseif name == :convergence
            name = :Convergence
        elseif name == :io
            name = Symbol("Input/Output")
        elseif name == :other
            name = :Other
        elseif name == :total
            name = :Total
        end
        return name
    end


    pretty_table(io, data; header = (["Each", "Relative", "Total"], [s, "Percentage", s_t]), 
                      row_labels = map(translate_for_table, flds),
                      formatters = (ft_printf("%3.4f", 1), ft_printf("%3.2f %%", 2), ft_printf("%3.4f", 3)),
                      title = title,
                      title_alignment = :c,
                      tf = table_formatter,
                      row_label_alignment = :l,
                      alignment = [:r, :r, :r],
                      body_hlines = [n-1],
                      row_label_column_title = "Timing type")
end

export read_results
"""
states, reports = read_results(pth; read_states = true, read_reports = true)

Read results from a given `output_path` provded to `simulate` or `simulator_config`.
"""
function read_results(pth;
        read_states = true,
        states = Vector{Dict{Symbol, Any}}(),
        read_reports = true,
        reports = [],
        range = nothing,
        name = nothing,
        verbose::Bool = true
    )
    indices = valid_restart_indices(pth)
    if isnothing(name)
        subpaths = splitpath(pth)
        name = subpaths[end]
    end
    if length(indices) == 0
        @error "Attempted to read simulated data from $pth, but no data was found."
        return (states, reports)
    elseif length(indices) != maximum(indices)
        @warn "Gap in dataset at $pth. Some outputs might end up empty."
    end
    if isnothing(range)
        range = 1:maximum(indices)
    end
    p = Progress(range[end]; enabled = verbose, desc = "Reading $name...")
    for i in range
        state, report = read_restart(pth, i; read_state = read_states, read_report = read_reports)
        if isnothing(report) && length(keys(state)) == 0
            break
        end
        if read_states
            push!(states, state)
        end
        if read_reports
            push!(reports, report)
        end
        next!(p)
    end
    return (states, reports)
end

function valid_restart_indices(pth)
    @assert isdir(pth)
    files = readdir(pth)
    indices = Vector{Int64}()
    for f in (files)
        if startswith(f, "jutul_")
            _, v = split(f, "jutul_")
            num, _ = splitext(v)
            push!(indices, parse(Int64, num))
        end
    end
    sort!(indices)
    return indices
end

function read_restart(pth, i; read_state = true, read_report = true)
    f = joinpath(pth, "jutul_$i.jld2")
    if isfile(f)
        state, report = jldopen(f, "r") do file
            if read_state
                state = file["state"]
            else
                state = Dict{Symbol, Any}()
            end
            if read_report
                report = file["report"]
            else
                report = nothing
            end
            stored_i = file["step"]
            if stored_i != i
                @warn "File contained step $stored_i, but was named as step $i."
            end
            return (state, report)
        end
    else
        state = Dict{Symbol, Any}()
        report = nothing
        @warn "Data for step $i was requested, but no such file was found."
    end
    return (state, report)
end

export report_timesteps, report_times
function report_timesteps(reports; ministeps = false, extra_out = false)
    if ministeps
        dt = Vector{Float64}()
        step_no = Vector{Int64}()
        for (i, r) = enumerate(reports)
            for m in r[:ministeps]
                if m[:success]
                    push!(dt, m[:dt])
                    push!(step_no, i)
                end
            end
        end
    else
        n = length(reports)
        dt = zeros(n)
        step_no = zeros(Int64, n)
        for (i, r) = enumerate(reports)
            t_loc = 0.0
            for m in r[:ministeps]
                if m[:success]
                    t_loc += m[:dt]
                end
            end
            dt[i] = t_loc
            step_no[i] = i
        end
    end
    if extra_out
        return (dt = dt, step = step_no)
    else
        return dt
    end
end

report_times(reports; ministeps = false) = cumsum(report_timesteps(reports, ministeps = ministeps, extra_out = false))


function get_cell_faces(N::AbstractMatrix{T}, nc = nothing) where T
    # Create array of arrays where each entry contains the faces of that cell
    if length(N) == 0
        cell_faces = ones(T, 1)
    else
        max_n = maximum(N)
        if isnothing(nc)
            nc = max_n
        else
            @assert max_n <= nc "Neighborship had maximum value of $max_n but number of cells provided was $nc: N = $N"
        end
        cell_faces = Vector{Vector{T}}(undef, nc)
        for i in 1:nc
            V = Vector{T}()
            sizehint!(V, 6)
            cell_faces[i] = V
        end
        for i in 1:size(N, 1)
            for j = 1:size(N, 2)
                push!(cell_faces[N[i, j]], j)
            end
        end
        # Sort each of them
        for i in cell_faces
            sort!(i)
        end
    end
    return cell_faces
end

function get_cell_neighbors(N, nc = maximum(N), includeSelf = true)
    # Find faces in each array
    t = typeof(N[1])
    cell_neigh = [Vector{t}() for i = 1:nc]
    for i in 1:size(N, 2)
        push!(cell_neigh[N[1, i]], N[2, i])
        push!(cell_neigh[N[2, i]], N[1, i])
    end
    # Sort each of them
    for i in eachindex(cell_neigh)
        loc = cell_neigh[i]
        if includeSelf
            push!(loc, i)
        end
        sort!(loc)
    end
    return cell_neigh
end

function get_facepos(N, arg...)
    if length(N) == 0
        t = eltype(N)
        faces = zeros(t, 0)
        facePos = ones(t, 2)
    else
        cell_faces = get_cell_faces(N, arg...)
        counts = [length(x) for x in cell_faces]
        facePos = cumsum([1; counts])
        faces = reduce(vcat, cell_faces)
    end
    return (faces, facePos)
end

export timing_breakdown
function timing_breakdown(report)
    D = Dict{Symbol, Any}(:total => report[:total_time])
    for ministep in report[:ministeps]
        mini_rep = timing_breakdown_ministep(ministep)
        for (k, v) in pairs(mini_rep)
            if haskey(D, k)
                D[k] += v
            else
                D[k] = v
            end
        end
    end
    return NamedTuple(pairs(D))
end

function timing_breakdown(report::NamedTuple; kwarg...)
    # Assume that it was already created.
    return report
end

function timing_breakdown(reports::Vector; reduce = true)
    avg = map(timing_breakdown, reports)
    if reduce
        next = Dict()
        for k in keys(avg[1])
            next[k] = sum(x -> x[k], avg)
        end
        avg = convert_to_immutable_storage(next)
    end
    return avg
end

function timing_breakdown_ministep(ministep)
    t_asm = 0.0
    t_solve = 0.0
    t_local = 0.0
    its = 0
    asm = 0
    for step in ministep[:steps]
        asm += 1
        t_asm += step[:secondary_time] + step[:equations_time] + step[:linear_system_time]
        if haskey(step, :linear_solve_time)
            its += 1
            t_solve += step[:linear_solve_time]
        end
        if haskey(step, :time_subdomains)
            t_local += step[:time_subdomains]
        end
    end
    return (assembly = t_asm, solve = t_solve, subdomains = t_local, its = its, no_asm = asm)
end


Base.@propagate_inbounds function Base.getindex(m::IndirectionMap, ix::Int)
    p = m.pos
    return view(m.vals, p[ix]:(p[ix+1]-1))
end

Base.length(m::IndirectionMap) = length(m.pos)-1

function Base.show(io::IO, t::MIME"text/plain", m::IndirectionMap)
    print(io, "IndirectionMap with $(length(m)) entities and total $(m.pos[end]-1) entries")
end

function get_mat_testgrid(name)
    base_path, = splitdir(pathof(Jutul))
    fn = joinpath(base_path, "..", "data", "testgrids", "$name.mat")
    return MAT.matread(fn)
end

export jutul_output_path
"""
    pth = jutul_output_path(name = missing; subfolder = "jutul", basedir = missing, create = true)

Get path for output. The final path will be found in /basedir/<subfolder/name.
If `subfolder=missing`, the path will be set to /basedir/name instead. `name`
will be autogenerated if not provided.

Pass the optional input `create = false` to avoid making the directory. To
globally set the default output dir, set `ENV["JUTUL_OUTPUT_PATH"]`` to your desired `basedir``.
"""
function jutul_output_path(name = missing; subfolder = "jutul", basedir = missing, create = true)
    if ismissing(basedir)
        default = "JUTUL_OUTPUT_PATH"
        if haskey(ENV, default)
            v = ENV[default]
            @assert !isfile(v) "basedir = $v cannot be created, file already exists with that name."
            basedir = v
        else
            basedir = tempdir()
        end
    end
    if ismissing(subfolder)
        subdir = basedir
    else
        subdir = joinpath(basedir, subfolder)
    end
    if create
        mkpath(subdir)
    end
    if ismissing(name)
        @assert isdir(basedir) "Missing name cannot be combined with create = false unless folder $subdir already exists."
        final_pth = tempname(subdir)
    else
        final_pth = joinpath(subdir, name)
    end
    if create
        mkpath(final_pth)
    end
    return final_pth
end


function get_step_report_errors(k::Symbol, step_reports)
    e = first(step_reports)[:errors][k]
    data = OrderedDict()
    for (ix, rep) in enumerate(step_reports)
        e = rep[:errors][k]
        for equation in e
            eq_label = equation.name
            for (crit_label, crit) in pairs(equation.criterions)
                t = join(crit.names, ", ")
                label = "$crit_label ($t)"
                sublabel = "$eq_label"
                key = (label, sublabel)

                local_error = crit.errors
                if length(local_error) == 1
                    local_error = only(local_error)
                end
                if haskey(data, key)
                    push!(data[key], local_error)
                else
                    data[key] = [local_error]
                end
            end
        end
    end
    data
end

"""
    merge_step_report_errors(data; fn = max)

Merge step reports errors of the same type using a pair wise reduction (default: max)
"""
function merge_step_report_errors(data; fn = max)
    new_data = similar(data[1])

    for d in data
        for (k, vals) in d
            if haskey(new_data, k)
                new_vals = new_data[k]
                @assert length(new_vals) == length(vals)
                for i in eachindex(new_vals)
                    new_vals[i] = fn.(new_vals[i], vals[i])
                end
                new_data[k] = new_vals
            else
                new_data[k] = copy(vals)
            end
        end
    end
    return new_data
end

function step_report_convergence_matrix(step_reports, groups = missing)
    if ismissing(groups)
        mkeys = keys(step_reports[1][:errors])
        groups = [(g, [g]) for g in mkeys]
    end
    n = 0
    m = length(step_reports)
    alldata = []
    for (gname, gkeys) in groups
        data = map(x -> get_step_report_errors(x, step_reports), gkeys)
        res = merge_step_report_errors(data)
        rkeys = keys(res)
        n += length(rkeys)
        push!(alldata, res)
    end
    data = Matrix{Any}(undef, m, n)
    ix = 1
    group_index = 1
    labels = String[]
    sublabels = String[]
    grouplabels = String[]
    for (gname, gkeys) in groups
        d = alldata[group_index]
        for (i, key) in enumerate(keys(d))
            names, equation = key
            push!(labels, names)
            push!(sublabels, equation)
            push!(grouplabels, "$gname")
            data[:, ix] .= d[key]
            ix += 1
        end
        group_index += 1
    end
    (data = data, names = labels, equations = sublabels, groups = grouplabels)
end

function print_step_report_convergence_matrix(step_reports, arg...; kwarg...)
    print_step_report_convergence_matrix!(stdout, step_reports, arg...; kwarg...)
end

function print_step_report_convergence_matrix!(io, step_reports, groups = missing; show_it = true, kwarg...)
    print_num(x) = @sprintf("%1.2e", x)
    function fmt(val::Base.AbstractVecOrTuple, i, j)
        join(map(print_num, val), " │ ")
    end
    function fmt(val::Real, i, j)
        print_num(val)
    end
    mat, names, equations, groups = step_report_convergence_matrix(step_reports, groups)
    subheader = map((x, y) -> "$x: $y", groups, equations)
    if show_it
        rl = 1:size(mat, 1)
    else
        rl = nothing
    end

    karg = (formatters = fmt, header = (names, subheader), alignment = :c, row_labels = rl, kwarg...)
    if io == stdout
        pretty_table(mat; karg...)
    else
        pretty_table(io, mat; karg...)
    end
end
