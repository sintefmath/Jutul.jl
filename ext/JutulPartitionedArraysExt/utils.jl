function to_new_indices(original_index, remapped_global_indices)
    return remapped_global_indices[original_index]
    # return findfirst(isequal(original_index), remapped_global_indices)
end

function from_new_indices(new_index, inverse_remapped_global_indices::Dict)
    return inverse_remapped_global_indices[new_index]
    # return findfirst(isequal(new_index), remapped_global_indices)
end

function remap_global_indices(p, np = max(p))
    counts = map(
        x -> sum(isequal(x), p),
        1:np
    )
    nc = length(p)
    remapped_global_indices = zeros(Int64, nc)
    for i in 1:np
        offset = process_offset(i, counts)
        self = findall(isequal(i), p)
        for (j, ix) in enumerate(self)
            remapped_global_indices[ix] = offset + j
        end
    end
    return (remapped_global_indices, counts)
end

function partition_boundary(N, p; np = max(p), nc = maximum(N))
    boundary = map(1:np) do i
        p_i = findall(isequal(i), p)
        bnd = Vector{Int}()
        interior_cells = BitArray(false for x = 1:nc)
        interior_cells[p_i] .= true

        for i in axes(N, 2)
            l = N[1, i]
            r = N[2, i]

            i_l = interior_cells[l]
            i_r = interior_cells[r]
            if i_l && !i_r
                push!(bnd, r)
            elseif i_r && !i_l
                push!(bnd, l)
            end
        end
        unique!(bnd)
    end
    return boundary
end

function distribute_case(case, full_partition, backend, ranks)
    model = case.model
    if model isa MultiModel
        model = case.model[full_partition.main_symbol]
    end
    domain = model.domain
    nc = number_of_cells(domain)
    N = get_neighborship(domain.representation)

    np = length(ranks)
    p = Jutul.main_partition(full_partition).partition

    # Create boundary map (original indices)
    boundary = partition_boundary(N, p, np = np, nc = nc)

    @assert length(p) == nc
    remapped_global_indices, counts = remap_global_indices(p, np)
    # @info "Remapped" remapped_global_indices counts
    tentative_partition = variable_partition(counts, nc)

    # Boundary in terms of the remapped indices used by the partitioner
    tentative_boundary = map(boundary) do b
        new_bnd = Vector{Int}()
        for i in b
            push!(new_bnd, to_new_indices(i, remapped_global_indices))
        end
        new_bnd
    end
    partition = combine_boundary(tentative_partition, tentative_boundary)

    # Extra mapping for degrees of freedom
    block_size = degrees_of_freedom_per_entity(model, Cells())
    if false
        @warn "Debug on" p boundary
        @info "?!" tentative_partition tentative_boundary partition
        @info "Part" remapped_global_indices p boundary
        @info "xtra" nc counts block_size
    end
    layout = matrix_layout(model.context)

    dof_partition = partition_degrees_of_freedom(p, boundary, remapped_global_indices, counts, nc, block_size, layout)
    partition = distribute_to_parray(partition, backend)

    tmp_mapper = Dict{Int, Int}()
    for (i, v) in enumerate(remapped_global_indices)
        tmp_mapper[v] = i
    end
    partition_original_indices = map(partition) do p
        p_original = similar(p)
        for (i, c) in enumerate(p)
            p_original[i] = from_new_indices(c, tmp_mapper) # This one is bad.
        end
        p_original
    end
    dof_partition = distribute_to_parray(dof_partition, backend)
    return (partition, dof_partition, counts, remapped_global_indices, partition_original_indices, block_size, nc)
end





function Jutul.executor_index_to_global(executor::PArrayExecutor, col_or_row, t::Symbol)
    return @inbounds executor.to_global[col_or_row]# executor.data[:to_global][col_or_row]
end

function distributed_ranks(b::PArrayBackend, np = 1)
    @assert np > 0
    return distribute_to_parray(LinearIndices((np,)), b)
end

function distributed_ranks(b::MPI_PArrayBackend, np = MPI.Comm_size(MPI.COMM_WORLD))
    @assert np == MPI.Comm_size(MPI.COMM_WORLD)
    ranks = distribute_to_parray(LinearIndices((np,)), b)
    return ranks
end

function distribute_to_parray(x, ::JuliaPArrayBackend)
    return x
end

function distribute_to_parray(x, ::MPI_PArrayBackend)
    @assert length(x) == MPI.Comm_size(MPI.COMM_WORLD)
    return distribute_with_mpi(x)
end

function distribute_to_parray(x, ::DebugPArrayBackend)
    return DebugArray(x)
end

function combine_boundary(tentative_partition, boundary)
    # @info "?" typeof(tentative_partition) typeof(boundary)
    boundary_owner = find_owner(tentative_partition, boundary)
    partition = map(union_ghost, tentative_partition, boundary, boundary_owner)
    # @info "ok" typeof(partition)
    return partition
end

function process_offset(i, counts)
    offset = 0
    for k in 1:(i-1)
        offset += counts[k]
    end
    return offset
end

function expand_boundary_to_dof(x, base_partition, remapped_indices, counts, nc, block_size, layout)
    # offset = block_size*sum(counts[1:(process_no-1)])
    p_dof = Vector{Int}()
    sizehint!(p_dof, block_size*nc)
    for i in x
        process_i = base_partition[i]
        offset_i = process_offset(process_i, counts)
        # This one should be local not global
        new_ix = to_new_indices(i, remapped_indices)
        r_i = new_ix - offset_i
        @assert !isnothing(r_i)
        n_local = counts[process_i]
        for b in 1:block_size
            mapped_ix = Jutul.alignment_linear_index(r_i, b, n_local, block_size, layout)
            global_ix = offset_i*block_size + mapped_ix
            # @info "$i -> $global_ix" new_ix mapped_ix n_local r_i offset_i process_i layout
            push!(p_dof, global_ix)
        end
    end
    p_dof = unique!(p_dof)
    @assert length(p_dof) == block_size*length(x)
    return p_dof
end

function partition_degrees_of_freedom(base_partition, boundary, remapped_indices, counts, nc, block_size, layout)
    tentative_dof_partition = variable_partition(block_size.*counts, block_size*nc)

    ix = 1:maximum(base_partition)

    dof_boundary = map(boundary, ix) do p_bnd, i
        expand_boundary_to_dof(p_bnd, base_partition, remapped_indices, counts, nc, block_size, layout)
    end
    # map(dof_boundary, tentative_dof_partition, base_partition, boundary, ix) do bnd, part, p_i, bnd_i, i
    #     @assert isempty(intersect(part, bnd)) "Failed: p=$part bnd=$bnd"
    # end
    return combine_boundary(tentative_dof_partition, dof_boundary)
end
