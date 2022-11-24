export SimpleHeatSystem

struct SimpleHeatSystem <: JutulSystem end

struct SimpleHeatEquation <: JutulEquation end

Base.@propagate_inbounds function update_equation_in_entity!(v, cellno, state, state0, eq::SimpleHeatEquation, model, dt, ldisc = local_discretization(eq, cellno))
    g = model.domain.grid
    g::CartesianMesh # Finite difference scheme based on structured cart mesh, put a type assert
    T = state.T
    T0 = state0.T

    function next(x, n)
        if x == n
            x⁺ = 1
        else
            x⁺ = x + 1
        end
        return x⁺
    end
    function prev(x, n)
        if x == 1
            x⁻ = n
        else
            x⁻ = x - 1
        end
        return x⁻
    end
    stencil_1d(T_c, T_l, T_r, h) = (T_l - 2*T_c + T_r)/h^2
    nx, ny = g.dims # Assume 2D grid
    i, j = cell_ijk(g, cellno)
    Δx, Δy, = cell_dims(g, cellno)
    # Temperature in cell center
    T_c = T[cellno]
    # Left-right (X)
    L = cell_index(g, (prev(i, nx), j))
    R = cell_index(g, (next(i, nx), j))
    ∂²x = stencil_1d(T_c, T[L], T[R], Δx)
    # Up-down (Y)
    U = cell_index(g, (i, prev(j, ny)))
    D = cell_index(g, (i, next(j, ny)))
    ∂²y = stencil_1d(T_c, T[U], T[D], Δy)
    # Now do the finite difference stencil, treating the dof at the inter-cell centers (staggered grid)
    ∂t = (T_c - T0[cellno])/dt
    v[] = ∂t - (∂²x + ∂²y)
end

number_of_equations_per_entity(model::SimulationModel, ::SimpleHeatEquation) = 1

local_discretization(eq::SimpleHeatEquation, i) = nothing

# Set up equation for system
function select_equations!(eqs, system::SimpleHeatSystem, model::SimulationModel)
    eqs[:heat_equation] = SimpleHeatEquation()
end

struct TVar <: ScalarVariable end

function select_primary_variables!(S, system::SimpleHeatSystem, model::SimulationModel)
    S[:T] = TVar()
end

minimum_value(::TVar) = 0
