function consolidate_distributed_results_on_disk!(pth, np, steps; cleanup = true)
    @assert isdir(pth)
    partitions = []
    for rank in 1:np
        proc_pth = rank_folder(pth, rank)
        part_path = joinpath(proc_pth, "partition.jld2")
        p = load(part_path)
        push!(partitions, p)
    end
    allpaths = []
    for step in steps
        step::Int
        states, reports, paths = read_step(pth, step, np)
        state, report = consolidate_distributed_results(states, reports, partitions)
        Jutul.write_result_jld2(pth, state, report, step)
        push!(allpaths, paths)
    end
    if cleanup
        # Delete the parallel results after consolidation.
        for paths in allpaths
            for path in paths
                rm(path)
            end
        end
    end
end

function read_step(pth, step, np)
    states = Vector{Any}(undef, np)
    reports = Vector{Any}(undef, np)
    paths = Vector{Any}(undef, np)
    for i in 1:np
        proc_pth = rank_folder(pth, i)
        states[i], reports[i] = Jutul.read_restart(proc_pth, step; read_state = true, read_report = true)
        paths[i] = joinpath(proc_pth, "jutul_$step.jld2")
    end
    return (states, reports, paths)
end

function rank_folder(pth, rank)
    joinpath(pth, "proc_$rank")
end

function consolidate_distributed_results(states, reports, partitions)
    state = Dict{Symbol, Any}()
    report = Dict{Symbol, Any}()
    mlabel = partitions[1]["main_partition_label"]
    if isnothing(mlabel)
        consolidate_cell_values!(state, states, partitions)
    else
        main_state = Dict{Symbol, Any}()
        for substate in states
            for (k, v) in pairs(substate)
                if k == mlabel
                    continue
                end
                @assert !haskey(state, k) "Inconsistent data - field $k is present in multiple ranks."
                state[k] = v
            end
        end
        consolidate_cell_values!(main_state, map(x -> x[mlabel], states), partitions)
        state[mlabel] = main_state
    end
    consolidate_report!(report, reports, partitions)
    return (state, report)
end

function consolidate_cell_values!(state, states, partitions)
    function replace_values!(nval::T, dval::T, p) where T<:AbstractVector
        for (v, g) in zip(dval, p)
            nval[g] = v
        end
    end
    function replace_values!(nval::T, dval::T, p) where T<:AbstractMatrix
        for (i, g) in enumerate(p)
            for j in axes(nval, 1)
                nval[j, g] = dval[j, i]
            end
        end
    end
    for (k, v) in first(states)
        partition = first(partitions)
        n = partition["n_total"]
        T = eltype(v)
        if v isa AbstractVector
            state[k] = Vector{T}(undef, n)
        elseif v isa AbstractMatrix
            m = size(v, 1)
            state[k] = Matrix{T}(undef, m, n)
        end
    end
    for (substate, partition) in zip(states, partitions)
        p = partition["partition"]
        for (k, v) in substate
            replace_values!(state[k], v, p)
        end
    end
end

function consolidate_report!(report, reports, partitions)
    for (k, v) in reports[1]
        # TODO: This should probably average up timings instead of taking the
        # first report only. But since most steps have some implicit barriers it
        # might be ok enough for most uses.
        report[k] = v
    end
end