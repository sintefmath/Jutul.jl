export ConservationLaw

struct ConservationLaw <: TervEquation
    accumulation::TervAutoDiffCache
    half_face_flux_cells::TervAutoDiffCache
    half_face_flux_faces::Union{TervAutoDiffCache,Nothing}
    flow_discretization::FlowDiscretization
end

function ConservationLaw(model, number_of_equations;
                            flow_discretization = model.domain.discretizations.mass_flow,
                            kwarg...)
    D = model.domain
    cell_unit = Cells()
    face_unit = Faces()
    nc = count_units(D, cell_unit)
    nhf = 2 * count_units(D, face_unit)

    face_partials = degrees_of_freedom_per_unit(model, face_unit)
    alloc = (n, unit) -> CompactAutoDiffCache(number_of_equations, n, model, unit = unit; kwarg...)
    acc = alloc(nc, cell_unit)
    hf_cells = alloc(nhf, cell_unit)
    if face_partials > 0
        hf_faces = alloc(nhf, face_unit)
    else
        hf_faces = nothing
    end
    ConservationLaw(acc, hf_cells, hf_faces, flow_discretization)
end

"Update positions of law's derivatives in global Jacobian"
function align_to_jacobian!(law::ConservationLaw, jac, model, u::Cells; equation_offset = 0, variable_offset = 0)
    fd = law.flow_discretization
    neighborship = model.domain.grid.neighborship

    acc = law.accumulation
    hflux_cells = law.half_face_flux_cells

    layout = matrix_layout(model.context)
    
    diagonal_alignment!(acc, jac, u, layout, target_offset = equation_offset, source_offset = variable_offset)
    half_face_flux_cells_alignment!(hflux_cells, acc, jac, layout, neighborship, fd, target_offset = equation_offset, source_offset = variable_offset)
end

function half_face_flux_cells_alignment!(face_cache, acc_cache, jac, layout, N, flow_disc; target_offset = 0, source_offset = 0)
    nu, ne, np = ad_dims(acc_cache)
    facepos = flow_disc.conn_pos
    nc = length(facepos) - 1
    Threads.@threads for cell in 1:nc
        @inbounds for f_ix in facepos[cell]:(facepos[cell + 1] - 1)
            f = flow_disc.conn_data[f_ix].face
            if N[1, f] == cell
                other = N[2, f]
            else
                other = N[1, f]
            end
            for e in 1:ne
                for d = 1:np
                    pos = find_jac_position(jac, cell + target_offset, other + source_offset, e, d, nu, nu, ne, np, layout)
                    set_jacobian_pos!(face_cache, f_ix, e, d, pos)
                end
            end
        end
    end
end

function align_to_jacobian!(law::ConservationLaw, jac, model, ::Faces; equation_offset = 0, variable_offset = 0)
    fd = law.flow_discretization
    neighborship = model.domain.grid.neighborship

    hflux_faces = law.half_face_flux_faces
    
    if !isnothing(hflux_faces)
        layout = matrix_layout(model.context)
        half_face_flux_faces_alignment!(hflux_faces, jac, layout, neighborship, fd, target_offset = equation_offset, source_offset = variable_offset)
    end
end


function half_face_flux_faces_alignment!(face_cache, jac, layout, N, flow_disc; target_offset = 0, source_offset = 0)
    nu, ne, np = ad_dims(face_cache)
    conn_data = flow_disc.conn_data
    for index in 1:length(conn_data)
        face = conn_data[index].face
        for e in 1:ne
            for d = 1:np
                pos = find_jac_position(jac, face + target_offset, face + source_offset, e, d, nu, nu, ne, np, layout)
                set_jacobian_pos!(face_cache, index, e, d, pos)
            end
        end
    end
end

function update_linearized_system_subset!(nz, r, model, law::ConservationLaw)
    acc = get_diagonal_cache(law)
    cell_flux = law.half_face_flux_cells
    face_flux = law.half_face_flux_faces
    cpos = law.flow_discretization.conn_pos

    @sync begin 
        @async update_linearized_system_subset_conservation_accumulation!(nz, r, model, acc, cell_flux, cpos)
        @async update_linearized_system_subset_cell_flux!(nz, model, acc, cell_flux, cpos)
        if !isnothing(face_flux)
            @async update_linearized_system_subset_face_flux!(nz, r, model, acc, face_flux, cpos)
        end
    end
end

function update_linearized_system_subset_conservation_accumulation!(nz, r, model, acc::CompactAutoDiffCache, cell_flux::CompactAutoDiffCache, conn_pos)
    nc, ne, np = ad_dims(acc)
    centries = acc.entries
    fentries = cell_flux.entries
    cp = acc.jacobian_positions
    Threads.@threads for cell = 1:nc
        for e in 1:ne
            diag_entry = get_entry(acc, cell, e, centries)
            @inbounds for i = conn_pos[cell]:(conn_pos[cell + 1] - 1)
                diag_entry -= get_entry(cell_flux, i, e, fentries)
            end

            @inbounds r[cell + nc*(e-1)] = diag_entry.value
            for d = 1:np
                apos = get_jacobian_pos(acc, cell, e, d, cp)
                @inbounds nz[apos] = diag_entry.partials[d]
            end
        end
    end
end

function update_linearized_system_subset_cell_flux!(Jz, model, acc::CompactAutoDiffCache, cell_flux::CompactAutoDiffCache, conn_pos)
    nc, ne, np = ad_dims(acc)
    fentries = cell_flux.entries
    fp = cell_flux.jacobian_positions
    Threads.@threads for cell = 1:nc
        @inbounds for i = conn_pos[cell]:(conn_pos[cell + 1] - 1)
            for e in 1:ne
                f = get_entry(cell_flux, i, e, fentries)
                for d = 1:np
                    df_di = f.partials[d]
                    fpos = get_jacobian_pos(cell_flux, i, e, d, fp)
                    @inbounds Jz[fpos] = df_di
                end
            end
        end
    end
end

function update_linearized_system_subset_face_flux!(jac, r, model, acc, face_flux, conn_pos)
    error("Not implemented yet")
end

function get_diagonal_cache(e::ConservationLaw)
    e.accumulation
end

function declare_pattern(model, e::ConservationLaw, ::Cells)
    df = e.flow_discretization
    hfd = Array(df.conn_data)
    n = number_of_units(model, e)
    # Fluxes
    I = map(x -> x.self, hfd)
    J = map(x -> x.other, hfd)
    # Diagonals
    D = [i for i in 1:n]

    I = vcat(I, D)
    J = vcat(J, D)

    return (I, J, n, n)
end

function declare_sparsity(model, e::ConservationLaw, ::Faces)
    @assert false "Not implemented."
end


#= function declare_sparsity(model, e::ConservationLaw)
    hfd = Array(model.domain.discretizations.KGrad.conn_data)
    n = number_of_units(model, e)
    # Fluxes
    I = map(x -> x.self, hfd)
    J = map(x -> x.other, hfd)
    # Diagonals
    D = [i for i in 1:n]

    I = vcat(I, D)
    J = vcat(J, D)

    nrows = number_of_equations_per_unit(e)
    ncols = number_of_partials_per_unit(e)
    nunits = number_of_units(model, e)
    if nrows > 1
        I = vcat(map((x) -> (x-1)*n .+ I, 1:nrows)...)
        J = repeat(J, nrows)
    end
    if ncols > 1
        I = repeat(I, ncols)
        J = vcat(map((x) -> (x-1)*n .+ J, 1:ncols)...)
    end
    n = number_of_equations(model, e)
    m = nunits*ncols
    return (I, J, n, m)
end =#
function convergence_criterion(model, storage, eq::ConservationLaw, r; dt = 1)
    n = number_of_equations_per_unit(eq)
    nc = number_of_cells(model.domain)
    pv = get_pore_volume(model)
    e = zeros(n)
    prm = storage.parameters
    for i = 1:n
        if haskey(prm, :ReferenceDensity)
            rhos = prm.ReferenceDensity[i]
        else
            rhos = 1
        end
        ri = @view r[(1:nc) .+ (i - 1) * nc]
        e[i] = mapreduce((pv, e) -> abs((dt/rhos) * e / pv), max, pv, ri)
    end
    return (e, 1.0)
end


function half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, neq, nder, equation_offset = 0, variable_offset = 0)
    n = size(fluxpos, 1)
    nf = size(fluxpos, 2)
    @assert nder == n / neq

    for i in 1:nf
        # Off diagonal positions
        cdi = conn_data[i]
        self = cdi.self
        other = cdi.other

        for col = 1:nder
            # Diagonal positions
            col_pos = (col - 1) * nc + other + variable_offset
            pos = jac.colptr[col_pos]:jac.colptr[col_pos + 1] - 1
            rowval = jac.rowval[pos]
            for row = 1:neq
                row_ix = self + (row - 1) * nc + equation_offset
                fluxpos[(row - 1) * nder + col, i] = pos[rowval .== row_ix][1]
                # @printf("Matching %d %d to %d\n", row_ix, col_pos, pos[rowval .== row_ix][1])
            end
        end
    end
end

function update_equation!(law::ConservationLaw, storage, model, dt)
    update_accumulation!(law, storage, model, dt)
    update_half_face_flux!(law, storage, model)
end

function get_diagonal_part(eq::ConservationLaw)
    return eq.accumulation
end
