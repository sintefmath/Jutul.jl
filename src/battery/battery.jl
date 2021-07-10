using Terv
export get_flow_volume

#########
# utils #
#########


function get_flow_volume(grid::MinimalECTPFAGrid)
    grid.volumes
end

function declare_units(G::MinimalECTPFAGrid)
    # Cells equal to number of pore volumes
    c = (unit = Cells(), count = length(G.volumes))
    # Faces
    f = (unit = Faces(), count = size(G.neighborship, 2))
    return [c, f]
end

################
# All EC-comps #
################

function single_unique_potential(
    model::SimulationModel{D, S}
    )where {D<:TervDomain, S<:ElectroChemicalComponent}
    return false
end

function degrees_of_freedom_per_unit(model, sf::ChargeAcc)
    return 1
end

function degrees_of_freedom_per_unit(model, sf::MassAcc)
    return 1
end

function degrees_of_freedom_per_unit(model, sf::KGrad)
    return 1
end

function degrees_of_freedom_per_unit(
    model::SimulationModel{D, S}, sf::Phi
    ) where {D<:TervDomain, S<:ElectroChemicalComponent}
    return 1
end

function number_of_units(model, pv::KGrad)
    """ Two fluxes per face """
    return 2*count_units(model.domain, Faces())
end

function number_of_units(model, ::Phi)
    return count_units(model.domain, Cells())
end

function number_of_units(model, ::C)
    return count_units(model.domain, Cells())
end

function number_of_units(model, ::T)
    return count_units(model.domain, Cells())
end


# ?Why not faces?
function associated_unit(::KGrad)
    Cells()
end

@inline function get_diagonal_cache(eq::Conservation)
    return eq.accumulation
end


function update_linearized_system_equation!(
    nz, r, model, law::Conservation
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
    law::Conservation, jac, model, u::Cells; equation_offset = 0, 
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

function declare_pattern(model, e::Conservation, ::Cells)
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

function declare_pattern(model, e::Conservation, ::Faces)
    df = e.flow_discretization
    cd = df.conn_data
    I = map(x -> x.self, cd)
    J = map(x -> x.face, cd)
    return (I, J)
end


############################
# Standard implementations #
############################

function get_conductivity(
    model::SimulationModel{D, S, F, C}
    ) where {D, S <: ElectroChemicalComponent, F, C}    
    return ones(number_of_units(model, Phi()))
end

function get_alpha(
    model::SimulationModel{D, S, F, Con}
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    return ones(number_of_units(model, C()))  * 100
    return repeat((10 .^ LinRange(3, -3, 10))', 10)'
end

function get_heat_cond(
    model::SimulationModel{D, S, F, Con}
    ) where {D, S <: ElectroChemicalComponent, F, Con}    
    return ones(number_of_units(model, T()))
end


@terv_secondary function update_as_secondary!(
    pot, tv::TPkGrad{Phi}, model::SimulationModel{D, S, F, C}, param, Phi
    ) where {D, S <: ElectroChemicalComponent, F, C}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    σ = get_conductivity(model)
    @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], Phi, σ)
end

@terv_secondary function update_as_secondary!(
    pot, tv::TPkGrad{C}, model::SimulationModel{D, S, F, Con}, param, C
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    α = get_alpha(model)
    @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], C, α)
end

@terv_secondary function update_as_secondary!(
    pot, tv::TPkGrad{T}, model::SimulationModel{D, S, F, Con}, param, T
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    # ? Sette til konstant??
    k = get_heat_cond(model)
    @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], T, k)
end


# ? Hva sker når man ganger med volume?
@terv_secondary function update_as_secondary!(
    acc, tv::MassAcc, model, param, C
    )
    V = get_flow_volume(model.domain.grid)
    @tullio acc[i] = C[i]
end

@terv_secondary function update_as_secondary!(
    acc, tv::EnergyAcc, model, param, T
    )
    V = get_flow_volume(model.domain.grid)
    @tullio acc[i] = T[i]
end
