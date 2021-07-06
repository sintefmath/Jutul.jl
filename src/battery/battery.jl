using Terv

#########
# utils #
#########


function get_flow_volume(grid::MinimalECTPFAGrid)
    grid.volumes
end

function build_forces(
    model::SimulationModel{G, S}; sources = nothing
    ) where {G<:TervDomain, S<:CurrentCollector}
    return (sources = sources,)
end

function declare_units(G::MinimalECTPFAGrid)
    # Cells equal to number of pore volumes
    c = (unit = Cells(), count = length(G.volumes))
    # Faces
    f = (unit = Faces(), count = size(G.neighborship, 2))
    return [c, f]
end

function degrees_of_freedom_per_unit(
    model::SimulationModel{D, S}, sf::Phi
    ) where {D<:TervDomain, S<:CurrentCollector}
    return 1 
end

function degrees_of_freedom_per_unit(
    model::SimulationModel{D, S}, sf::Conductivity
    ) where {D<:TervDomain, S<:CurrentCollector}
    return 1
end

function degrees_of_freedom_per_unit(model, sf::TotalCharge)
    return 1
end

function degrees_of_freedom_per_unit(model, sf::TPFlux)
    return 1
end

function minimum_output_variables(
    system::CurrentCollector, primary_variables
    )
    [:TotalCharge]
end

function single_unique_potential(
    model::SimulationModel{D, S}
    )where {D<:TervDomain, S<:CurrentCollector}
    return false
end

function initialize_variable_value!(
    state, model, pvar::Conductivity, symb::Symbol, val::Number
    )
    n = values_per_unit(model, pvar)
    return initialize_variable_value!(
        state, model, pvar, symb, repeat([val], n)
        )
end

function default_value(v::Conductivity)
    return 1.0
end

function number_of_units(model, pv::TPFlux)
    """ Two fluxes per face """
    return 2*count_units(model.domain, Faces())
end

# ?Why not faces?
function associated_unit(::TPFlux)
    Cells()
end

function update_linearized_system_equation!(
    nz, r, model, law::ChargeConservation
    )
    
    acc = get_diagonal_cache(law)
    cell_flux = law.half_face_flux_cells
    cpos = law.flow_discretization.conn_pos

    begin 
        update_linearized_system_subset_conservation_accumulation!(nz, r, model, acc, cell_flux, cpos)
        fill_equation_entries!(nz, nothing, model, cell_flux)
    end
end


function align_to_jacobian!(
    law::ChargeConservation, jac, model, u::Cells; equation_offset = 0, 
    variable_offset = 0
    )
    fd = law.flow_discretization
    neighborship = get_neighborship(model.domain.grid)

    acc = law.accumulation
    hflux_cells = law.half_face_flux_cells
    diagonal_alignment!(
        acc, jac, u, model.context, target_offset = equation_offset, 
        source_offset = variable_offset)
    half_face_flux_cells_alignment!(
        hflux_cells, acc, jac, model.context, neighborship, fd, 
        target_offset = equation_offset, source_offset = variable_offset
        )
end

function declare_pattern(model, e::ChargeConservation, ::Cells)
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

    return (I, J)
end

function declare_pattern(model, e::ChargeConservation, ::Faces)
    df = e.flow_discretization
    cd = df.conn_data
    I = map(x -> x.self, cd)
    J = map(x -> x.face, cd)
    return (I, J)
end


#############
# Variables #
#############

function select_primary_variables_system!(
    S, domain, system::ElectroChemicalComponent, formulation
    )
    S[:Phi] = Phi()
end

function select_secondary_variables_flow_type!(
    S, domain, system, formulation, flow_type::ChargeFlow
    )
    S[:TPFlux] = TPFlux()
    S[:TotalCharge] = TotalCharge()
end

function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation
    )
    eqs[:charge_conservation] = (ChargeConservation, 1)
end

# @terv_secondary function update_as_secondary!(
#     totcharge, tv::TotalCharge, model::SimulationModel{G, S}, param
#     ) where {G, S<:CurrentCollector}
#     @tullio totcharge[i] = 0 # Charge neutrality
# end


@terv_secondary function update_as_secondary!(
    pot, tv::TPFlux, model, param, Phi
    )
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio pot[i] = half_face_two_point_grad(conn_data[i], Phi)
end

@terv_secondary function update_as_secondary!(
    pot, tv::Phi, model, param, Phi
    )
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    context = model.context
    update_cell_neighbor_potential_cc!(
        pot, conn_data, Phi, context, kernel_compatibility(context)
        )
end

function update_cell_neighbor_potential_cc!(
    dpot, conn_data, phi, context, ::KernelDisallowed
    )
    Threads.@threads for i in eachindex(conn_data)
        c = conn_data[i]
        @inbounds dpot[phno] = half_face_two_point_grad(
                c.self, c.other, c.T, phi
        )
    end
end

function update_cell_neighbor_potential_cc!(
    dpot, conn_data, phi, context, ::KernelAllowed
    )
    @kernel function kern(dpot, @Const(conn_data))
        ph, i = @index(Global, NTuple)
        c = conn_data[i]
        dpot[ph] = half_face_two_point_grad(c.self, c.other, c.T, phi)
    end
    begin
        d = size(dpot)
        kernel = kern(context.device, context.block_size, d)
        event_jac = kernel(dpot, conn_data, phi, ndrange = d)
        wait(event_jac)
    end
end
