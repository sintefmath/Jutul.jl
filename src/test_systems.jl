export ScalarTestSystem, ScalarTestDomain, ScalarTestForce
export XVar

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end

function number_of_cells(::ScalarTestDomain) 1 end

function get_domain_intersection(u::TervUnit, target_d::ScalarTestDomain, source_d::ScalarTestDomain)
    # This domain always interacts with the single cell in instances of itself, and nothing else
    return (1, 1, Cells())
end

# Driving force for the test equation
struct ScalarTestForce 
    value
end

# Equations
struct ScalarTestEquation <: DiagonalEquation
    equation
    function ScalarTestEquation(G::TervDomain, npartials::Integer; context = DefaultContext(), kwarg...)
        nc = number_of_cells(G)
        @assert nc == 1 # We use nc for clarity of the interface - but it should always be one!
        ne = 1 # Single, scalar equation
        e = CompactAutoDiffCache(ne, nc, npartials, context = context; kwarg...)
        new(e)
    end
end

function declare_sparsity(model, e::ScalarTestEquation)
    return (1, 1, 1, 1)
end

# Model features
function allocate_equations!(eqs, storage, model::SimulationModel{G, S}; kwarg...) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    eqs[:TestEquation] = ScalarTestEquation(model.domain, 1, context = model.context; kwarg...)
end

function update_equation!(eq::ScalarTestEquation, storage, model, dt)
    X = storage.state.XVar
    X0 = storage.state0.XVar
    equation = get_entries(eq)
    @. equation = (X - X0)/dt
end

function build_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    return (sources = sources,)
end

function apply_forces_to_equation!(storage, model, eq::ScalarTestEquation, force::ScalarTestForce)
    equation = get_entries(eq)
    @. equation -= force.value
end

function update_cross_term!(ct::InjectiveCrossTerm, eq::ScalarTestEquation, target_storage, source_storage, target, source, dt)
    X_T = target_storage.state.XVar
    X_S = source_storage.state.XVar
    function f(X_S, X_T)
        X_T - X_S
    end
    # Source term with AD context from source model - will end up as off-diagonal block
    @. ct.crossterm_source = f(X_S, value(X_T))
    # Source term with AD context from target model - will be inserted into equation
    @. ct.crossterm_target = f(value(X_S), X_T)
end

struct XVar <: ScalarPrimaryVariable end


function select_primary_variables(domain, system::ScalarTestSystem, formulation)
    return [XVar()]
end
