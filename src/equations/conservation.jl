struct ConservationLaw <: TervEquation
    accumulation::AbstractArray
    half_face_flux::AbstractArray
    accumulation_jac_pos::AbstractArray   # Usually diagonal entries
    half_face_flux_jac_pos::AbstractArray # Equal length to half face flux
end

function ConservationLaw(G::TervDomain, nder::Integer = 0; jacobian_row_offset = 0, context = DefaultContext(), equations_per_unit = 1)
    I = index_type(context)
    nu = equations_per_unit
    # Create conservation law for a given grid with a number of partials
    nc = number_of_cells(G)
    nf = number_of_half_faces(G)

    # Will be filled in later
    accpos = zeros(I, nu*nder, nc)
    fluxpos = zeros(I, nu*nder, nf)

    accpos = transfer(context, accpos)
    fluxpos = transfer(context, fluxpos)

    ConservationLaw(nc, nf, accpos, fluxpos, nder, context = context, equations_per_unit = equations_per_unit)
end

function ConservationLaw(nc::Integer, nhf::Integer, 
                         accpos::AbstractArray, fluxpos::AbstractArray, 
                         npartials::Integer = 0; context = DefaultContext(), equations_per_unit = 1)
    acc = allocate_array_ad(equations_per_unit, nc, context = context, npartials = npartials)
    flux = allocate_array_ad(equations_per_unit, nhf, context = context, npartials = npartials)
    ConservationLaw(acc, flux, accpos, fluxpos)
end

"Update positions of law's derivatives in global Jacobian"
function align_to_linearized_system!(law::ConservationLaw, lsys::LinearizedSystem, model)
    # Note: jacobian_row_offset needs to be added somewhere for multiphase
    jac = lsys.jac
    # Note: We copy this back to host if it is on GPU to avoid rewriting these functions for CuArrays
    conn_data = Array(model.domain.conn_data)
    accpos = Array(law.accumulation_jac_pos)
    fluxpos = Array(law.half_face_flux_jac_pos)

    @show accpos
    neq = number_of_equations_per_unit(law)
    nder = number_of_partials_per_unit(law)
    nc = size(law.accumulation, 2)
    accumulation_sparse_pos!(accpos, jac, neq, nder)
    half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, neq, nder)

    law.accumulation_jac_pos .= accpos
    law.half_face_flux_jac_pos .= fluxpos
end

function update_linearized_system!(lsys::LinearizedSystem, model, law::ConservationLaw)
    G = model.domain
    context = model.context
    ker_compat = kernel_compatibility(context)
    apos = law.accumulation_jac_pos
    neq = number_of_equations_per_unit(law)
    jac = lsys.jac
    r = lsys.r
    # Fill in diagonal
    # @info "Accumulation fillin"
    fill_accumulation!(jac, r, law.accumulation, apos, neq, context, ker_compat)
    # Fill in off-diagonal
    fpos = law.half_face_flux_jac_pos
    # @info "Half face flux fillin"
    fill_half_face_fluxes(jac, r, G.conn_pos, law.half_face_flux, apos, fpos, neq, context, ker_compat)
end

function number_of_equations_per_unit(e::ConservationLaw)
    return size(e.half_face_flux, 1)
end

function number_of_partials_per_unit(e::ConservationLaw)
    return size(e.accumulation_jac_pos, 1) ÷ number_of_equations_per_unit(e)
end

function declare_sparsity(model, e::ConservationLaw)
    hfd = Array(model.domain.conn_data)
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
    if nrows > 1
        I = vcat(map((x) -> (x-1)*n .+ I, 1:nrows)...)
        J = repeat(J, nrows)
    end
    if ncols > 1
        I = repeat(I, ncols)
        J = vcat(map((x) -> (x-1)*n .+ J, 1:ncols)...)
    end

    return (I, J)
end

function convergence_criterion(model, storage, eq::ConservationLaw, lsys::LinearizedSystem; dt = 1)
    n = number_of_equations_per_unit(eq)
    nc = number_of_cells(model.domain)
    pv = model.domain.pv
    e = zeros(n)
    for i = 1:n
        e[i] = mapreduce((pv, e) -> abs(dt*e/pv), max, pv, lsys.r[(1:nc) .+ (i-1)*nc])
    end
    return (e, 1.0)
end

function accumulation_sparse_pos!(accpos, jac, neq, nder)
    n = size(accpos, 1)
    nc = size(accpos, 2)
    @assert nder == n/neq
    for i in 1:nc
        for col = 1:nder
            # Diagonal positions
            col_pos = (col-1)*nc + i
            pos = jac.colptr[col_pos]:jac.colptr[col_pos+1]-1
            for row = 1:neq
                row_ix = (row-1)*nc + i
                accpos[(row-1)*nder + col, i] = pos[jac.rowval[pos] .== row_ix][1]
            end
        end
    end
end

function half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, neq, nder)
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
            col_pos = (col-1)*nc + other
            pos = jac.colptr[col_pos]:jac.colptr[col_pos+1]-1
            rowval = jac.rowval[pos]
            for row = 1:neq
                row_ix = self + (row-1)*nc
                fluxpos[(row-1)*nder + col, i] = pos[rowval .== row_ix][1]
                # @printf("Matching %d %d to %d\n", row_ix, col_pos, pos[rowval .== row_ix][1])
            end
        end
    end
end
