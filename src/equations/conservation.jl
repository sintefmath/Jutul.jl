export ConservationLaw

struct ConservationLaw <: TervEquation
    accumulation::TervAutoDiffCache
    half_face_flux_cells::TervAutoDiffCache
    half_face_flux_faces::Union{TervAutoDiffCache, Nothing}
    function ConservationLaw(nc, nhf, neqs, cell_partials, face_partials = 0; cell_unit = Cells(), face_unit = Faces(), kwarg...)
        alloc = (n, np, unit) -> CompactAutoDiffCache(n, np, unit = unit; kwarg...)
        acc = alloc(nc, cell_partials, cell_unit)
        hf_cells = alloc(nhf, cell_partials, cell_unit)
        if face_partials > 0
            hf_faces = alloc(nhf, face_partials, face_unit)
        else
            hf_faces = nothing
        end
        new(acc, hf_cells, hf_faces)
    end
end

function ConservationLaw(model, number_of_equations; cell_unit = Cells(), face_unit = Faces(), kwarg...)
    D = model.domain
    nc = count_units(D, cell_unit)
    nhf = 2*count_units(D, face_unit)

    cell_partials = degrees_of_freedom_per_unit(model, cell_unit)
    face_partials = degrees_of_freedom_per_unit(model, face_unit)

    ConservationLaw(nc, nhf, number_of_equations, cell_partials, face_partials, cell_unit = cell_unit, face_unit = cell_unit; kwarg...)
end

"Update positions of law's derivatives in global Jacobian"
function align_to_jacobian!(law::ConservationLaw, jac, model; row_offset = 0, col_offset = 0)
    # Note: We copy this back to host if it is on GPU to avoid rewriting these functions for CuArrays
    conn_data = Array(model.domain.discretizations.KGrad.conn_data)
    accpos = Array(law.accumulation_jac_pos)
    fluxpos = Array(law.half_face_flux_jac_pos)

    neq = number_of_equations_per_unit(law)
    nder = number_of_partials_per_unit(law)
    nc = size(law.accumulation, 2)
    accumulation_sparse_pos!(accpos, jac, neq, nder, row_offset, col_offset)
    half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, neq, nder, row_offset, col_offset)

    law.accumulation_jac_pos .= accpos
    law.half_face_flux_jac_pos .= fluxpos
end

function update_linearized_system_subset!(jac, r, model, law::ConservationLaw)
    cpos = model.domain.discretizations.KGrad.conn_pos
    context = model.context
    ker_compat = kernel_compatibility(context)
    apos = law.accumulation_jac_pos
    neq = number_of_equations_per_unit(law)
    # Fill in diagonal
    # @info "Accumulation fillin"
    fill_accumulation!(jac, r, law.accumulation, apos, neq, context, ker_compat)
    # Fill in off-diagonal
    fpos = law.half_face_flux_jac_pos
    # @info "Half face flux fillin"
    fill_half_face_fluxes(jac, r, cpos, law.half_face_flux, apos, fpos, neq, context, ker_compat)
end

function number_of_equations_per_unit(e::ConservationLaw)
    return size(e.half_face_flux, 1)
end

function number_of_partials_per_unit(e::ConservationLaw)
    return length(e.accumulation[1].partials)
end

function declare_sparsity(model, e::ConservationLaw)
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
end

function convergence_criterion(model, storage, eq::ConservationLaw, r; dt = 1)
    n = number_of_equations_per_unit(eq)
    nc = number_of_cells(model.domain)
    pv = get_pore_volume(model)
    e = zeros(n)
    for i = 1:n
        e[i] = mapreduce((pv, e) -> abs(dt*e/pv), max, pv, r[(1:nc) .+ (i-1)*nc])
    end
    return (e, 1.0)
end

function accumulation_sparse_pos!(accpos, jac, neq, nder, row_offset = 0, col_offset = 0)
    n = size(accpos, 1)
    nc = size(accpos, 2)
    @assert nder == n/neq
    for i in 1:nc
        for col = 1:nder
            # Diagonal positions
            col_pos = (col-1)*nc + i + col_offset
            pos = jac.colptr[col_pos]:jac.colptr[col_pos+1]-1
            for row = 1:neq
                row_ix = (row-1)*nc + i + row_offset
                accpos[(row-1)*nder + col, i] = pos[jac.rowval[pos] .== row_ix][]
            end
        end
    end
end

function half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, neq, nder, row_offset = 0, col_offset = 0)
    n = size(fluxpos, 1)
    nf = size(fluxpos, 2)
    @assert nder == n/neq

    for i in 1:nf
        # Off diagonal positions
        cdi = conn_data[i]
        self = cdi.self
        other = cdi.other

        for col = 1:nder
            # Diagonal positions
            col_pos = (col-1)*nc + other + col_offset
            pos = jac.colptr[col_pos]:jac.colptr[col_pos+1]-1
            rowval = jac.rowval[pos]
            for row = 1:neq
                row_ix = self + (row-1)*nc + row_offset
                fluxpos[(row-1)*nder + col, i] = pos[rowval .== row_ix][1]
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
