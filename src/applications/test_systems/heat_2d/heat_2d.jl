export SimpleHeatSystem

struct SimpleHeatSystem <: JutulSystem end

struct SimpleHeatEquation <: JutulEquation end

function update_equation_in_entity!(v, cellno, state, state0, eq::SimpleHeatEquation, model, dt, ldisc = local_discretization(eq, cellno))
    g = model.domain.grid
    g::CartesianMesh # Finite difference scheme based on structured cart mesh, put a type assert
    T = state.T
    T0 = state0.T

    function stencil_1d(T, T_c, c, l, r, h, N)
        # Absorbing boundary conditions - just subsitute the value of U so that it always matches
        if c == 1
            T_l = T_c
        else
            T_l = T[l]
        end
        if c == N
            T_r = T_c
        else
            T_r = T[r]
        end
        return (T_l - 2*T_c + T_r)/h^2
    end

    nx, ny = g.dims # Assume 2D grid
    i, j = cell_ijk(g, cellno)
    Δx, Δy, = cell_dims(g, cellno)
    # Temperature in cell center
    T_c = T[cellno]
    # Left-right (X)
    L = cell_index(g, (i-1, j))
    R = cell_index(g, (i+1, j))
    ∂²x = stencil_1d(T, T_c, i, L, R, Δx, nx)
    # Up-down (Y)
    U = cell_index(g, (i, j-1))
    D = cell_index(g, (i, j+1))
    ∂²y = stencil_1d(T, T_c, j, U, D, Δy, ny)
    # Now do the finite difference stencil, treating the dof at the inter-cell centers (staggered grid)
    ∂t = (T_c - T0[cellno])/dt
    v[] = ∂t + ∂²x + ∂²y
end

number_of_equations_per_entity(::SimpleHeatEquation) = 1

# Set up equation for system
function select_equations_system!(eqs, domain, system::SimpleHeatSystem, formulation)
    eqs[:heat_equation] = SimpleHeatEquation()
end

struct TVar <: ScalarVariable end

function select_primary_variables!(S, domain, system::SimpleHeatSystem, formulation)
    S[:T] = TVar()
end

minimum_value(::TVar) = 0
