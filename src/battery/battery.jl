using Terv
export ElectroChemicalComponent, CurrentCollector
export get_test_setup_battery, get_cc_grid


### Classes and corresponding overwritten functions

abstract type ElectroChemicalComponent <: TervSystem end
struct CurrentCollector <: ElectroChemicalComponent end
abstract type ElectroChemicalGrid <: TervGrid end
struct Phi <: ScalarVariable end
struct Conductivity <: ComponentVariable end
struct TotalCharge <: GroupedVariables end
# New version of CellNeighborPotentialDifference
struct TPFlux <: GroupedVariables end


# Instead of DarcyMassMobilityFlow
#? Is this the correct substitution
struct ChargeFlow <: FlowType end
include_face_sign(::ChargeFlow) = false

struct ChargeConservation <: TervEquation
    accumulation::TervAutoDiffCache
    accumulation_symbol::Symbol
    half_face_flux_cells::TervAutoDiffCache
    half_face_flux_faces::Union{TervAutoDiffCache,Nothing}
    flow_discretization::FlowDiscretization
end

function ChargeConservation(
    model, number_of_equations;
    flow_discretization = model.domain.discretizations.charge_flow,
    accumulation_symbol = :TotalCharge,
    kwarg...
    )

    D = model.domain
    cell_unit = Cells()
    face_unit = Faces()
    nc = count_units(D, cell_unit)
    nf = count_units(D, face_unit)
    nhf = 2 * nf
    face_partials = degrees_of_freedom_per_unit(model, face_unit)

    alloc = (n, unit, n_units_pos) -> CompactAutoDiffCache(
    number_of_equations, n, model, unit = unit, n_units_pos = n_units_pos,
    context = model.context; kwarg...
    )

    acc = alloc(nc, cell_unit, nc)
    hf_cells = alloc(nhf, cell_unit, nhf)

    if face_partials > 0
        hf_faces = alloc(nf, face_unit, nhf)
    else
        hf_faces = nothing
    end

    ChargeConservation(
        acc, accumulation_symbol, hf_cells, hf_faces, flow_discretization
    )
end

# TPFA grid
"Minimal struct for TPFA-like grid. Just connection data and pore-volumes"
struct MinimalECTPFAGrid{R<:AbstractFloat, I<:Integer} <: ElectroChemicalGrid
    pore_volumes::AbstractVector{R}
    neighborship::AbstractArray{I}
    function MinimalECTPFAGrid(pv, N)
        nc = length(pv)
        pv::AbstractVector
        @assert size(N, 1) == 2  "Two neighbors per face"
        if length(N) > 0
            @assert minimum(N) > 0   "Neighborship entries must be positive."
            @assert maximum(N) <= nc "Neighborship must be limited to number of cells."
        end
        @assert all(pv .> 0)     "Pore volumes must be positive"
        new{eltype(pv), eltype(N)}(pv, N)
    end
end

# ??How do you implement sources??
function build_forces(
    model::SimulationModel{G, S}; sources = nothing
    ) where {G<:TervDomain, S<:CurrentCollector}
    return (sources = sources,)
end

function declare_units(G::MinimalECTPFAGrid)
    # Cells equal to number of pore volumes
    c = (unit = Cells(), count = length(G.pore_volumes))
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
    face_flux = law.half_face_flux_faces
    cpos = law.flow_discretization.conn_pos

    @sync begin 
        @async update_linearized_system_subset_conservation_accumulation!(nz, r, model, acc, cell_flux, cpos)
        @async fill_equation_entries!(nz, nothing, model, cell_flux)
        if !isnothing(face_flux)
            conn_data = law.flow_discretization.conn_data
            @async update_linearized_system_subset_face_flux!(nz, model, face_flux, cpos, conn_data)
        end
    end
end


"Update positions of law's derivatives in global Jacobian"
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
    # S[:Conductivity] = Conductivity()
end

function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation
    )
    eqs[:charge_conservation] = (ChargeConservation, 1)
end


@terv_secondary function update_as_secondary!(
    totcharge, tv::TotalCharge, model::SimulationModel{G, S}, param
    ) where {G, S<:CurrentCollector}
    @tullio totcharge[i] = 0 # Charge neutrality
end


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


### PHYSICS

@inline function half_face_two_point_grad(
    conn_data::NamedTuple, p::AbstractArray
    )
    half_face_two_point_grad(
        conn_data.self, conn_data.other, conn_data.T, p
        )
end

@inline function half_face_two_point_grad(
    c_self::I, c_other::I, T, phi::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    return -T * (phi[c_self] - value(phi[c_other]))
end

####

## From flux.jl

function update_equation!(law::ChargeConservation, storage, model, dt)
    update_accumulation!(law, storage, model, dt)
    update_intrinsic_sources!(law, storage, model, dt)
    update_half_face_flux!(law, storage, model, dt)
end

function update_intrinsic_sources!(
    law::ChargeConservation, storage, model, dt
    )
    # Do nothing
end

function update_half_face_flux!(
    law::ChargeConservation, storage, model, dt
    )
    fd = law.flow_discretization
    update_half_face_flux!(law, storage, model, dt, fd)
end

function update_half_face_flux!(
    law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ChargeFlow}

    pot = storage.state.TPFlux  # ?WHy is this named pot?
    flux = get_entries(law.half_face_flux_cells)
    update_fluxes_from_potential!(flux, pot)
end

function update_fluxes_from_potential!(flux, pot)
    @tullio flux[i] = pot[i]
end


@inline function get_diagonal_cache(eq::ChargeConservation)
    return eq.accumulation
end


##############
# Main funcs #
##############

function get_test_setup_battery()
    G = get_cc_grid()
    timesteps = [1.0, 2.0]

    sys = CurrentCollector()
    model = SimulationModel(G, sys, context = DefaultContext())
    forces = build_forces(model)

    # State is dict with pressure in each cell
    phi = 1.
    init = Dict(:Phi => phi)
    state0 = setup_state(model, init)
    state0[:Phi][1] = 2  # Endrer startverdien, skal ikke endre svaret
    
    # Model parameters
    parameters = setup_parameters(model)

    return (state0, model, parameters, forces, timesteps)
end


function get_cc_grid(
    perm = nothing, poro = nothing, volumes = nothing, extraout = false
    )
    name = "pico"
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."

    N = exported["G"]["faces"]["neighbors"]
    N = Int64.(N)
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')
        
    # Cells
    cell_centroids = copy((exported["G"]["cells"]["centroids"])')
    # Faces
    face_centroids = copy((exported["G"]["faces"]["centroids"][internal_faces, :])')
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    if isnothing(perm)
        perm = copy((exported["rock"]["perm"])')
        perm = ones(size(perm)) # Endre permeability til 1 
    end

    # Deal with cell data
    if isnothing(poro)
        poro = vec(exported["rock"]["poro"])
    end
    if isnothing(volumes)
        volumes = vec(exported["G"]["cells"]["volumes"])
    end
    pv = poro.*volumes

    @debug "Data unpack complete. Starting transmissibility calculations."
    # Deal with face data
    T_hf = compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    T = compute_face_trans(T_hf, N)

    G = MinimalECTPFAGrid(pv, N)
    z = nothing
    g = nothing

    ft = ChargeFlow()
    # ??Hva gjør SPU og TPFA??
    flow = TwoPointPotentialFlow(SPU(), TPFA(), ft, G, T, z, g)
    disc = (charge_flow = flow,)
    D = DiscretizedDomain(G, disc)

    if extraout
        return (D, exported)
    else
        return D
    end
end
