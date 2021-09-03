export ScalarTestSystem, ScalarTestDomain, ScalarTestForce
export XVar

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end

function number_of_cells(::ScalarTestDomain) 1 end

function get_domain_intersection(u::TervUnit, target_d::ScalarTestDomain, source_d::ScalarTestDomain, target_symbol, source_symbol)
    # This domain always interacts with the single cell in instances of itself, and nothing else
    (target = [1], source = [1], target_unit = Cells(), source_unit = Cells())
end

# Driving force for the test equation
struct ScalarTestForce
    value
end

# Equations
struct ScalarTestEquation <: DiagonalEquation
    equation
    function ScalarTestEquation(model, npartials::Integer; context = DefaultContext(), kwarg...)
        D = model.domain
        nc = number_of_cells(D)
        @assert nc == 1 # We use nc for clarity of the interface - but it should always be one!
        ne = 1 # Single, scalar equation
        e = CompactAutoDiffCache(ne, nc, npartials, context = context; kwarg...)
        new(e)
    end
end

function declare_sparsity(model, e::ScalarTestEquation, layout)
    return SparsePattern(1, 1, 1, 1, layout)
end


function select_equations_system!(eqs, domain, system::ScalarTestSystem, formulation)
    eqs[:test_equation] = (ScalarTestEquation, 1)
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

function apply_forces_to_equation!(storage, model, eq::ScalarTestEquation, force::ScalarTestForce, time)
    equation = get_entries(eq)
    @. equation -= force.value
end

function update_cross_term!(ct::InjectiveCrossTerm, eq::ScalarTestEquation, target_storage, source_storage, target_model, source_model, target, source, dt)
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

struct XVar <: ScalarVariable end

function select_primary_variables!(S, domain, system::ScalarTestSystem, formulation)
    S[:XVar] = XVar()
end
