
function Jutul.PArraySimulator(case::JutulCase, full_partition::Jutul.AbstractDomainPartition; comm = MPI.COMM_WORLD, backend = JuliaPArrayBackend(), kwarg...)
    data = JutulStorage()
    for (k, v) in kwarg
        data[k] = v
    end
    main_part = Jutul.main_partition(full_partition)
    np = maximum(main_part.partition)
    ranks = distributed_ranks(backend, np)
    tmr = PTimer(ranks)
    data[:global_timer] = tmr
    data[:verbose] = i_am_main(ranks)
    data[:number_of_processes] = np

    partition, dof_partition, counts, remapped_ix, partition_original_indices, block_size, nc = distribute_case(case, full_partition, backend, ranks)

    data[:partition] = partition
    data[:dof_partition] = dof_partition
    data[:counts] = counts
    data[:block_size] = block_size
    data[:nc] = nc
    data[:comm] = comm

    data[:recorder] = ProgressRecorder()

    # @info "Starting" main_part.partition

    # map(partition_original_indices, full_partition.subsets, ranks) do p_buf, p, i
    #     missing_part = setdiff(p, p_buf)
    #     @info "Block $i" p_buf p
    #     @warn "Overlap?" missing_part
    # end
    # Make simulators
    n_owned = 0
    process_start = typemax(Int)
    representative_model = missing
    simulators = map(partition_original_indices, ranks) do p, i
        # @info "$i" p
        n_self = counts[i]
        n_owned += n_self
        process_start = min(process_start, process_offset(i, counts))
        exec = PArrayExecutor(backend, i, remapped_ix[p], n_self = n_self, partition = p, n_total = sum(counts), comm = comm)
        (; model, state0, parameters) = case
        # Boundary padding has been added, update local subset
        missing_part = setdiff(main_part.subsets[i], p)
        @assert length(missing_part) == 0
        if true
            main_part.subsets[i] = p
            m = submodel(model, full_partition, i)
        else
            m = submodel(model, p)
        end
        s0 = substate(state0, model, m, :variables)
        prm = substate(parameters, model, m, :parameters)
        sim = Simulator(m, state0 = s0, parameters = prm, executor = exec)
        if ismissing(representative_model)
            representative_model = m
        end
        sim
    end
    data[:nc_process] = n_owned
    data[:process_offset] = process_start
    data[:simulators] = simulators
    data[:distributed_residual] = pzeros(dof_partition)
    data[:distributed_cell_buffer] = pzeros(partition)
    data[:distributed_residual_buffer] = pzeros(dof_partition)
    data[:model] = representative_model

    return PArraySimulator(backend, data)
end

function Jutul.preprocess_forces(psim::PArraySimulator, forces)
    simulators = psim.storage[:simulators]
    forces_per_step = forces isa AbstractVector
    inner_forces = map(simulators) do sim
        if forces_per_step
            f = map(x -> subforces(x, sim.model), forces)
        else
            f = subforces(forces, sim.model)
        end
        f
    end
    return (forces = inner_forces, forces_per_step = forces_per_step)
end

function Jutul.simulate_parray(case::JutulCase, partition, backend::PArrayBackend; kwarg...)
    (; dt, forces) = case
    outer_sim = PArraySimulator(case, partition, backend = backend)
    result = simulate!(outer_sim, dt, forces = forces)
    # states = consolidate_distributed_states(model, result.states, outer_sim.storage.partition, outer_sim.storage.nc)
    PartitionedArrays.print_timer(outer_sim.storage.global_timer, linechars = :ascii)
    return result
end

function Jutul.partition_distributed(N, w;
    comm = MPI.COMM_WORLD,
    nc = maximum(vec(N)),
    np,
    kwarg...
    )
    root = 0
    if MPI.Comm_rank(comm) == root
        p = Jutul.partition(N, np, w; kwarg...)
        @assert length(p) == nc "Expected length of partition to be $nc, was $(length(p))."
    else
        p = Array{Int}(undef, nc)
    end

    MPI.Bcast!(p, root, comm)
    return p
end


