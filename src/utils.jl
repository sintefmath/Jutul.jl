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
    get_matrix_view(v, n, m, transp, offset)
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
                for (i, e) in enumerate(Array(local_errors))
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
                            mstr = "from model "*String(model)
                        else
                            mstr = ""
                        end
                        worst_name = "$(eq.name) ($(local_names[i])) $mstr"
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
        s  = ":"
    elseif count_crit == count_ok
        s = " ✔️"
    else
        worst_print = @sprintf "%2.3e (ϵ = %2.3e)" worst_val*worst_tol worst_tol
        s = ". Worst value:\n\t - $worst_name at $worst_print."
    end
    @info "$(id)It. $iteration/$max_its: $count_ok/$count_crit criteria converged$s"

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
        h1 = Highlighter(f = not_converged,
                         crayon = crayon"red" )
    
        function nearly_converged(data, i, j)
            if j == rpos
                d = data[i, j]
                t = tols[i]
                return d > t && d < nearly_factor*t
            else
                return false
            end
        end
        h2 = Highlighter(f = nearly_converged,
                         crayon = crayon"yellow")
    
        function converged(data, i, j)
            if j == rpos
                return data[i, j] <= tols[i]
            else
                return false
            end
        end
        h3 = Highlighter(f = converged,
                         crayon = crayon"green")
    
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

function report_stats(reports)
    total_time = 0
    # Counts
    total_its = 0
    total_linearizations = 0
    # Same, but for wasted (iterations that were part of a cut time-step)
    wasted_its = 0
    wasted_linearizations = 0
    wasted_linear_iterations = 0
    # Various timings
    total_finalize = 0
    total_assembly = 0
    total_linear_update = 0
    total_linear_solve = 0
    total_linear_iterations = 0
    total_update = 0
    total_convergence = 0
    total_io = 0

    total_steps = length(reports)
    total_ministeps = 0
    for outer_rep in reports
        total_time += outer_rep[:total_time]
        if haskey(outer_rep, :output_time)
            t_io = outer_rep[:output_time]
            total_io += t_io
            total_time += t_io
        end
        for mini_rep in outer_rep[:ministeps]
            total_ministeps += 1
            if haskey(mini_rep, :finalize_time)
                total_finalize += mini_rep[:finalize_time]
            end

            s = stats_ministep(mini_rep[:steps])
            total_linearizations += s.linearizations
            total_its += s.newtons
            total_linear_update += s.linear_system
            total_linear_solve += s.linear_solve
            total_linear_iterations += s.linear_iterations
            total_update += s.update_time
            total_assembly += s.assembly
            total_convergence += s.convergence

            if !mini_rep[:success]
                wasted_its += s.newtons
                wasted_linearizations += s.linearizations
                wasted_linear_iterations += s.linear_iterations
            end
        end
    end
    sum_measured = total_assembly + total_linear_update + total_linear_solve + total_update + total_convergence + total_io
    other_time = total_time - sum_measured
    totals = (
                assembly = total_assembly,
                linear_system = total_linear_update,
                linear_solve = total_linear_solve,
                update_time = total_update,
                convergence = total_convergence,
                io = total_io,
                other = other_time,
                total = total_time
            )
    
    n = total_its
    m = total_linearizations
    l = total_linear_iterations
    linscale = v -> v / max(m, 1)
    itscale = v -> v / max(n, 1)
    each = (
                assembly = linscale(total_assembly),
                linear_system = linscale(total_linear_update),
                linear_solve = itscale(total_linear_solve),
                update_time = itscale(total_update),
                convergence = linscale(total_convergence),
                io = total_io/total_ministeps,
                other = itscale(other_time),
                total = itscale(total_time)
            )
    return (
            newtons = total_its,
            linearizations = total_linearizations,
            linear_iterations = total_linear_iterations,
            wasted = (newtons = wasted_its,
                      linearizations = wasted_linearizations,
                      linear_iterations = wasted_linear_iterations),
            steps = total_steps,
            ministeps = total_ministeps,
            time_sum = totals,
            time_each = each
           )
end

function stats_ministep(reports)
    local_linearizations = 0
    local_its = 0
    local_assembly = 0
    local_convergence = 0
    local_linear_update = 0
    update = 0
    linsolve = 0
    linear_iterations = 0
    for rep in reports
        local_linearizations += 1
        if haskey(rep, :update_time)
            local_its += 1
            update += rep[:update_time]
            linsolve += rep[:linear_solve_time]
            linear_iterations += rep[:linear_iterations]
        end
        local_assembly += rep[:assembly_time]
        local_linear_update += rep[:linear_system_time]
        local_convergence += rep[:convergence_time]
    end
    return (linearizations = local_linearizations,
            newtons = local_its, 
            assembly = local_assembly,
            convergence = local_convergence,
            update_time = update,
            linear_iterations = linear_iterations,
            linear_solve = linsolve,
            linear_system = local_linear_update)
end

function pick_time_unit(t)
    m = maximum(t)
    units = [(24*3600, "Days"), (3600, "Hours"), (60, "Minutes"), (1, "Seconds"), (1e-3, "Milliseconds"), (1e-6, "Microseconds"), (1e-9, "Nanoseconds")]
    for u in units
        if m > u[1]
            return u
        end
    end
    # Fallback
    return (1, "Seconds")
end

function print_stats(reports::AbstractArray; kwarg...)
    stats = report_stats(reports)
    print_stats(stats; kwarg...)
end

function print_stats(stats; kwarg...)
    print_iterations(stats; kwarg...)
    print_timing(stats; kwarg...)
end

function print_iterations(stats; title = "Number of iterations", table_formatter = :tf_unicode)
    flds = [:newtons, :linearizations, :linear_iterations]
    names = [:Newtons, :Linearizations, Symbol("Linear solver its.")]
    data = Array{Any}(undef, length(flds), 5)
    nstep = stats.steps
    nmini = stats.ministeps

    tot_time = stats.time_sum.total
    time = map(f -> tot_time/stats[f], flds)
    u, s = pick_time_unit(time)

    for (i, f) in enumerate(flds)
        raw = stats[f]
        data[i, 1] = raw/nstep         # Avg per step
        data[i, 2] = raw/nmini         # Avg per mini
        data[i, 3] = time[i]/u         # Time each
        data[i, 4] = stats[:wasted][f] # Wasted total
        data[i, 5] = raw               # Total
    end

    
    pretty_table(data; header = (["Avg/step", "Avg/ministep", "Time per", "Wasted", "Total"], ["$nstep steps", "$nmini ministeps", s, "", ""]), 
                      row_names = names,
                      title = title,
                      title_alignment = :c,
                      row_name_alignment = :l,
                      tf = table_formatter,
                      formatters = (ft_printf("%3.4f", 3)),
                      row_name_column_title = "Type")
end

function print_timing(stats; title = "Simulator timing", table_formatter = :tf_unicode)
    flds = collect(keys(stats.time_each))
    
    n = length(flds)
    hl_last = Highlighter(f = (data, i, j) -> i == n, crayon = Crayon(background = :light_blue))
    
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

    # hl = Highlighter((data, i, j) -> (i == length(flds)), crayon"fg:bold");
    # highlighters = hl,
    function translate_for_table(name)
        if name == :assembly
            name = :Properties
        elseif name == :linear_system
            name = :Assembly
        elseif name == :linear_solve
            name = Symbol("Linear solve")
        elseif name == :update_time
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


    pretty_table(data; header = (["Each", "Fraction", "Total"], [s, "Percent", s_t]), 
                      row_names = map(translate_for_table, flds),
                      formatters = (ft_printf("%3.4f", 1), ft_printf("%3.2f %%", 2), ft_printf("%3.4f", 3)),
                      title = title,
                      title_alignment = :c,
                      tf = table_formatter,
                      row_name_alignment = :l,
                      alignment = [:r, :r, :r],
                      body_hlines = [n-1],
                      row_name_column_title = "Name")
end

export read_results
"""
Read results from a given output_path provded to simulate or simulator_config
"""
function read_results(pth; read_states = true, states = Vector{Dict{Symbol, Any}}(),
                           read_reports = true, reports = [], range = nothing, name = nothing, verbose::Bool = true)
    indices = valid_restart_indices(pth)
    if isnothing(name)
        subpaths = splitpath(pth)
        name = subpaths[end]
    end
    if length(indices) != maximum(indices)
        @warn "Gap in dataset. Some outputs might end up empty."
    end
    if isnothing(range)
        range = 1:maximum(indices)
    end
    p = Progress(range[end]; enabled = verbose, desc = "Reading $name...")
    for i in range
        state, report = read_restart(pth, i; read_state = read_states, read_report = read_reports)
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


function get_cell_faces(N, nc = nothing)
    # Create array of arrays where each entry contains the faces of that cell
    t = eltype(N)
    if length(N) == 0
        cell_faces = ones(t, 1)
    else
        if isnothing(nc)
            nc = maximum(N)
        end
        cell_faces = [Vector{t}() for i = 1:nc]
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
