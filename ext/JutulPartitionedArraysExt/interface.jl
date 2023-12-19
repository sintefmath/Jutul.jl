
function Jutul.PArraySimulator(case::JutulCase, full_partition::Jutul.AbstractDomainPartition;
        backend = JuliaPArrayBackend(),
        order = :default,
        simulator_constructor = (m; kwarg...) -> Simulator(m; kwarg...),
        primary_buffer = false,
        kwarg...
        )
    data = JutulStorage()
    for (k, v) in kwarg
        data[k] = v
    end
    comm = backend_communicator(backend)
    main_part = Jutul.main_partition(full_partition)
    main_label = Jutul.main_partition_label(full_partition)
    np = maximum(main_part.partition)
    ranks = distributed_ranks(backend, np)
    data[:verbose] = i_am_main(ranks)
    data[:is_main_process] = i_am_main(ranks)
    data[:number_of_processes] = np

    # TODO: Deal with this monsterous interface.
    partition, dof_partition, counts, remapped_ix, partition_original_indices, block_size, nc = distribute_case(case, full_partition, backend, ranks, order = order)

    data[:partition] = partition
    data[:dof_partition] = dof_partition
    data[:counts] = counts
    data[:block_size] = block_size
    data[:nc] = nc
    data[:comm] = comm

    data[:recorder] = ProgressRecorder()

    # Make simulators
    n_owned = 0
    process_start = typemax(Int)
    representative_model = missing
    simulators = map(partition_original_indices, ranks) do p, i
        # @info "$i" p
        n_self = counts[i]
        n_owned += n_self
        process_start = min(process_start, process_offset(i, counts))
        exec = PArrayExecutor(backend, i, remapped_ix[p],
            main_label = main_label,
            n_self = n_self,
            number_of_processes = np,
            partition = p,
            n_total = sum(counts),
            comm = comm
        )
        (; model, state0, parameters) = case
        # Boundary padding has been added, update local subset
        missing_part = setdiff(main_part.subsets[i], p)
        @assert length(missing_part) == 0
        main_part.subsets[i] = p
        m = submodel(model, full_partition, i)
        s0 = substate(state0, model, m, :variables)
        prm = substate(parameters, model, m, :parameters)
        sim = simulator_constructor(m, state0 = s0, parameters = prm, executor = exec)
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
    data[:distributed_solution_buffer] = pzeros(dof_partition)
    data[:model] = representative_model
    data[:main_label] = main_label
    if primary_buffer
        main_model, = get_main_model(representative_model, main_label)
        pvar_def = (; pairs(main_model.primary_variables)...)
        e = missing
        for (k, v) in pairs(pvar_def)
            e_v = Jutul.associated_entity(v)
            if ismissing(e)
                e = e_v
            else
                @assert e == e_v "Distributed primary variables assuems that main model have same associated entity for primary variables ($e != $e_v)"
            end
        end
        T_primary = Jutul.scalarized_primary_variable_type(main_model, pvar_def)
        pvar_buf = pzeros(T_primary, partition)
        data[:distributed_primary_variables] = pvar_buf
        data[:distributed_primary_variables_def] = pvar_def
    end
    return PArraySimulator(backend, data)
end

function Jutul.simulator_executor(sim::PArraySimulator)
    # We do not return a PArrayExecutor since that is reserved for the inner
    # simulators. The outer simulator should act like the default.
    return Jutul.default_executor()
end

function get_main_model(model, l, storage = nothing)
    if model isa MultiModel
        model = model[l]
        if !isnothing(storage)
            storage = storage[l]
        end
    end
    return (model, storage)
end

function Jutul.MPI_PArrayBackend(; comm = MPI.COMM_WORLD)
    # Add a constructor that actually supports MPI.
    MPI.Init()
    Jutul.MPI_PArrayBackend(comm)
end

backend_communicator(b::Jutul.MPI_PArrayBackend) = b.comm
backend_communicator(::Any) = MPI.COMM_WORLD


function Jutul.preprocess_forces(psim::PArraySimulator, forces)
    simulators = psim.storage[:simulators]
    forces_per_step = forces isa AbstractVector
    inner_forces = map(simulators) do sim
        m = Jutul.get_simulator_model(sim)
        function F(x)
            sf = subforces(x, m)
            preprocessed = Jutul.preprocess_forces(sim, sf)
            return first(preprocessed)
        end
        if forces_per_step
            f = map(F, forces)
        else
            f = F(forces)
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

function Jutul.partition_distributed(N, edge_weights, node_weights = missing;
        comm = MPI.COMM_WORLD,
        nc = maximum(vec(N)),
        np,
        partitioner = MetisPartitioner(),
        groups = [Int[]],
        kwarg...
    )
    root = 0
    if MPI.Comm_rank(comm) == root
        if ismissing(node_weights)
            node_weights = fill(1, nc)
            for group in groups
                for g in group
                    node_weights[g] = 10
                end
            end
        end
        p = Jutul.partition_hypergraph(N, np, partitioner;
            num_nodes = nc,
            edge_weights = edge_weights,
            node_weights = node_weights,
            groups = groups,
            kwarg...
        )
        p = Int.(p)
        @assert length(p) == nc "Expected length of partition to be $nc, was $(length(p))."
    else
        p = Array{Int}(undef, nc)
    end

    MPI.Bcast!(p, root, comm)
    return p
end

function Jutul.parray_synchronize_primary_variables(psim::PArraySimulator; update_secondary = true)
    @assert haskey(psim.storage, :distributed_primary_variables)
    primary_buffer = psim.storage[:distributed_primary_variables]
    pvar_def = psim.storage[:distributed_primary_variables_def]
    simulators = psim.storage[:simulators]
    l = psim.storage[:main_label]

    map(simulators, local_values(primary_buffer)) do sim, pvar_vec
        m, s = get_main_model(Jutul.get_simulator_model(sim), l, Jutul.get_simulator_storage(sim))
        Jutul.scalarize_primary_variables!(pvar_vec, m, s.primary_variables, pvar_def)
    end
    consistent!(primary_buffer) |> wait

    map(simulators, local_values(primary_buffer)) do sim, pvar_vec
        model, s = get_main_model(Jutul.get_simulator_model(sim), l, Jutul.get_simulator_storage(sim))
        Jutul.descalarize_primary_variables!(s.primary_variables, model, pvar_vec)

        if update_secondary
            # TODO: NB assumes that all secondary variables are cell wise.
            n_own = sim.executor.data[:n_self]
            n_total = length(pvar_vec)
            state = s.state
            # We own the first n_own values and these need not be updated. The
            # ghost cells are now synchronized for primary and the dependent
            # secondary variables in those cells should also be updated.
            ghost_ix = (n_own+1):n_total
            for (k, v) in pairs(model.secondary_variables)
                Jutul.update_secondary_variable!(state[k], v, model, state, ghost_ix)
            end
        end
    end
end
