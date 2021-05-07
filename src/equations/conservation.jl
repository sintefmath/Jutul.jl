struct ConservationLaw <: TervEquation
    accumulation::AbstractArray
    half_face_flux::AbstractArray
    accumulation_jac_pos::AbstractArray   # Usually diagonal entries
    half_face_flux_jac_pos::AbstractArray # Equal length to half face flux
end

function ConservationLaw(G::TervDomain, lsys, nder::Integer = 0; jacobian_row_offset = 0, context = DefaultContext(), equations_per_unit = 1)
    F = float_type(context)
    I = index_type(context)
    nu = equations_per_unit
    # Create conservation law for a given grid with a number of partials
    nc = number_of_cells(G)
    nf = number_of_half_faces(G)

    accpos = zeros(I, nu*nder, nc)
    fluxpos = zeros(I, nu*nder, nf)
    # Note: jacobian_row_offset needs to be added somewhere for multiphase
    jac = lsys.jac
    # Note: We copy this back to host if it is on GPU to avoid rewriting these functions for CuArrays
    conn_data = Array(G.conn_data)
    accumulation_sparse_pos!(accpos, jac, nu, nder)
    half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, nu, nder)

    # Once positions are figured out (in a CPU context since Jacobian is not yet transferred)
    # we copy over the data to target device (or do nothing if we are on CPU)
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

function accumulation_sparse_pos!(accpos, jac, nu, nder)
    n = size(accpos, 1)
    nc = size(accpos, 2)
    @assert nder == n/nu
    for i in 1:nc
        for col = 1:nder
            # Diagonal positions
            col_pos = (col-1)*nc + i
            pos = jac.colptr[col_pos]:jac.colptr[col_pos+1]-1
            for row = 1:nu
                row_ix = (row-1)*nc + i
                accpos[(row-1)*nder + col, i] = pos[jac.rowval[pos] .== row_ix][1]
            end
        end
    end
end

function half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, nu, nder)
    n = size(fluxpos, 1)
    nf = size(fluxpos, 2)
    @assert nder == n/nu

    for i in 1:nf
        # Off diagonal positions
        cd = conn_data[i]
        self = cd.self
        other = cd.other

        for col = 1:nder
            # Diagonal positions
            col_pos = (col-1)*nc + other
            pos = jac.colptr[col_pos]:jac.colptr[col_pos+1]-1
            rowval = jac.rowval[pos]
            for row = 1:nu
                row_ix = self + (row-1)*nc
                fluxpos[(row-1)*nder + col, i] = pos[rowval .== row_ix][1]
                # @printf("Matching %d %d to %d\n", row_ix, col_pos, pos[rowval .== row_ix][1])
            end
        end
    end
end
