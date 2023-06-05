function unit_diagonalize!(r, J::SparseMatrixCSC, n_self)
    T = eltype(J)
    rows = rowvals(J)
    nzval = nonzeros(J)
    for k in eachindex(rows)
        row = rows[k]
        if row > n_self
            nzval[k] = zero(T)
        end
    end
    for i in (n_self+1:length(r))
        r[i] = zero(eltype(r))
        J[i, i] = one(T)
    end
end

function unit_diagonalize!(r, J::Jutul.StaticSparsityMatrixCSR, n_self)
    for i in (n_self+1:length(r))
        r[i] = zero(eltype(r))
    end
    T = eltype(J)
    cols = Jutul.colvals(J)
    nzval = nonzeros(J)
    for row in (n_self+1:length(r))
        for k in nzrange(J, row)
            col = cols[k]
            if col == row
                v = -one(T)
            else
                v = zero(T)
            end
            nzval[k] = v
        end
    end
end

function setup_parray_mul!(simulators, ix = nothing)
    operators = map(simulators) do sim
        lsys = sim.storage.LinearizedSystem
        if !isnothing(ix)
            lsys = lsys[ix, ix]
        end
        Jutul.linear_operator(lsys)
    end
    function distributed_mul!(Y::PVector, X::PVector, α, β)
        @tic "communication" consistent!(X) |> wait
        map(local_values(Y), local_values(X), operators) do y, x, local_op
            mul!(y, local_op, x, α, β)
            nothing
        end
        @tic "communication" consistent!(Y) |> wait
        return Y
    end
    return (Y, X, α, β) -> distributed_mul!(Y::PVector, X::PVector, α, β)
end

function parray_update_preconditioners_outer!(sim::PArraySimulator, prec_def)
    preconditioner_base, preconditioners = prec_def
    storage = sim.storage
    recorder = storage.recorder
    # Additional function call for dispatch nesting
    return Jutul.parray_update_preconditioners!(sim, preconditioner_base, preconditioners, recorder)
end

function Jutul.parray_update_preconditioners!(sim, preconditioner_base, preconditioners, recorder)
   map(sim.storage.simulators, preconditioners) do sim, prec
        sys = sim.storage.LinearizedSystem
        model = sim.model
        storage = sim.storage
        Jutul.update_preconditioner!(prec, sys, model, storage, recorder, sim.executor)
        prec
    end
    return (preconditioner_base, preconditioners)
end


function Jutul.parray_preconditioner_apply!(Y, main_prec, X, preconditioners, simulator, arg...)
    X::PVector
    Y::PVector
    @tic "communication" consistent!(X) |> wait
    map(local_values(Y), local_values(X), preconditioners, ghost_values(X)) do y, x, prec, x_g
        @. x_g = 0.0
        apply!(y, prec, x, arg...)
    end
    @tic "communication" consistent!(Y) |> wait
    Y
end

function Jutul.parray_linear_system_operator(simulators, b)
    distributed_mul! = setup_parray_mul!(simulators)
    n = length(b)
    return LinearOperator(Float64, n, n, false, false, distributed_mul!)
end

function parray_preconditioner_linear_operator(simulator, lsolve, b)
    @tic "precond" main_prec, preconditioners = parray_update_preconditioners_outer!(simulator, lsolve.preconditioner)
    prec_apply! = (Y, X, arg...) -> Jutul.parray_preconditioner_apply!(Y::PVector, main_prec, X::PVector, preconditioners, simulator, arg...)
    n = length(b)
    M = LinearOperator(Float64, n, n, false, false, prec_apply!)
    return Jutul.PrecondWrapper(M)
end


function LinearAlgebra.axpy!(α, x::PVector, y::PVector)
    # y = x*a + y
    @boundscheck @assert PartitionedArrays.matching_local_indices(axes(x,1),axes(y,1))
    n = length(x)
    # consistent!(x) |> wait
    # consistent!(y) |> wait

    if n != length(y)
        throw(DimensionMismatch("x has length $n, but y has length $(length(y))"))
    end
    map(local_values(x), local_values(y)) do x_i, y_i
        LinearAlgebra.axpy!(α, x_i, y_i)
        nothing
    end
    # consistent!(y) |> wait
    y
end

function LinearAlgebra.axpby!(α, x::PVector, β, y::PVector)
    # y = x*a + y*b
    @boundscheck @assert PartitionedArrays.matching_local_indices(axes(x,1),axes(y,1))
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x has length $n, but y has length $(length(y))"))
    end
    map(local_values(x), local_values(y)) do x_i, y_i
        LinearAlgebra.axpby!(α, x_i, β, y_i)
        nothing
    end
    y
end

