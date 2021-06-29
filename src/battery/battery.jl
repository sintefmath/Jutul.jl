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

function build_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:TervDomain, S<:CurrentCollector}
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

function minimum_output_variables(
    system::CurrentCollector, primary_variables
    )
    [:TotalCharge]
end


# To get right number of dof for CellNeigh...
function single_unique_potential(
    model::SimulationModel{D, S}
    )where {D<:TervDomain, S<:CurrentCollector}
    return false
end

function select_secondary_variables_flow_type!(
    S, domain, system, formulation, flow_type::ChargeFlow
    )
end

# function initialize_variable_value!(state, model, pvar::Conductivity, symb::Symbol, val::Number)
#     V = repeat([val], number_of_units(model, pvar))
#     return initialize_variable_value!(state, model, pvar, symb, V)
# end

# ??Hvorfor denne??
function initialize_variable_value!(
    state, model, pvar::Conductivity, symb::Symbol, val::Number
    )
    n = values_per_unit(model, pvar)
    return initialize_variable_value!(state, model, pvar, symb, repeat([val], n))
end


function default_value(v::Conductivity)
    return 1.0
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


# Selection of variables
function select_primary_variables_system!(S, domain, system::ElectroChemicalComponent, formulation)
    S[:Phi] = Phi()
end

### Fra variable_evaluation

# ? What is rholambda
@terv_secondary function update_as_secondary!(
    ρλ_i, tv::Conductivity, model, param
    )
end

@terv_secondary function update_as_secondary!(
    pot, tv::Phi, model, param, Phi
    ) #should resisitvity be added?
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    context = model.context
    if mf.gravity
        update_cell_neighbor_potential_cc!(
            pot, conn_data, Phi, context, kernel_compatibility(context)
            )
    else
        update_cell_neighbor_potential_cc!(
            pot, conn_data, Phi, context, kernel_compatibility(context)
            )
    end
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


function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation
    )
    eqs[:charge_conservation] = (ChargeConservation, 1)
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

### PHYSICS

@inline function half_face_two_point_grad(
    c_self::I, c_other::I, T, phi::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    return -T*two_point_potential_drop_half_face(c_self, c_other, phi)
end

@inline function two_point_potential_drop_half_face(
    c_self, c_other, phi::AbstractVector
    )
    return two_point_potential_drop(phi[c_self], value(phi[c_other]))
end


@inline function two_point_potential_drop(phi_self::Real, phi_other::Real)
    return phi_self - phi_other
end

@inline function half_face_two_point_flux_fused(
    c_self::I, c_other::I, T, phi::AbstractArray{R}, λ::AbstractArray{R},
    ) where {R<:Real, I<:Integer}
    
    θ = half_face_two_point_grad(c_self, c_other, T, phi)
    λᶠ = spu_upwind(c_self, c_other, θ, λ)
    return λᶠ*θ
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

    # ?? Can i just skip this? What does it normally do??
    # pot = storage.state.CellNeighborPotentialDifference
    # mob = storage.state.MassMobilities
    # flux = get_entries(law.half_face_flux_cells)
    # conn_data = law.flow_discretization.conn_data
    # update_fluxes_from_potential!(flux, conn_data, pot, mob)
end

# function update_fluxes_from_potential!(flux, conn_data, pot, mob)
#     @tullio flux[phno, i] = spu_upwind(conn_data[i].self, conn_data[i].other, pot[phno, i], view(mob, phno, :))
# end



@inline function get_diagonal_cache(eq::ChargeConservation)
    return eq.accumulation
end


# TODO: Må ha rktig secondary variables, i tillegg til tilhørende dependency, som definieres via update_as_secondary

function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::ChargeFlow)
    # S[:CellNeighborPotentialDifference] = CellNeighborPotentialDifference()
    S[:TotalCharge] = TotalCharge()
    S[:Conductivity] = Conductivity()
end


###


function get_test_setup_battery(context = "cpu", timesteps = [1.0, 2.0], pvfrac = 0.05)
    G = get_cc_grid()

    nc = number_of_cells(G)
    pv = G.grid.pore_volumes
    timesteps = timesteps*3600*24

    if context == "cpu"
        context = DefaultContext()
    elseif isa(context, String)
        error("Unsupported target $context")
    end
    @assert isa(context, TervContext)

    # Parameters
    phi = 1

    sys = CurrentCollector()
    model = SimulationModel(G, sys, context = context)

    # ? Can i just skip this??
    # ? I think this is where resisitvity would go
    # s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)


    # System state
    tot_time = sum(timesteps)
    irate = pvfrac*sum(pv)/tot_time
    src = [SourceTerm(1, irate), 
        SourceTerm(nc, -irate)]
    forces = build_forces(model, sources = src)

    # State is dict with pressure in each cell
    init = Dict(:Phi => phi)
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)

    state0 = setup_state(model, init)
    return (state0, model, parameters, forces, timesteps)
end


function get_cc_grid(perm = nothing, poro = nothing, volumes = nothing, extraout = false)
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
    if size(cell_centroids, 1) == 3
        z = cell_centroids[3, :]
        g = gravity_constant
    else
        z = nothing
        g = nothing
    end

    ft = ChargeFlow()
    flow = TwoPointPotentialFlow(SPU(), TPFA(), ft, G, T, z, g)
    disc = (charge_flow = flow,)
    D = DiscretizedDomain(G, disc)

    if extraout
        return (D, exported)
    else
        return D
    end
end
