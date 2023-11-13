
"""
ILU(0) preconditioner on CPU
"""
mutable struct ILUZeroPreconditioner <: JutulPreconditioner
    factor
    dim
    left::Bool
    right::Bool
    function ILUZeroPreconditioner(; left = true, right = false)
        @assert left || right "Left or right preconditioning must be enabled or it will have no effect."
        new(nothing, nothing, left, right)
    end
end

is_left_preconditioner(p::ILUZeroPreconditioner) = p.left
is_right_preconditioner(p::ILUZeroPreconditioner) = p.right

function set_dim!(ilu, A, b)
    T = eltype(b)
    if T<:AbstractFloat
        d = 1
    else
        d = length(T)
    end
    ilu.dim = d .* size(A)
end

function update_preconditioner!(ilu::ILUZeroPreconditioner, A, b, context, executor)
    if isnothing(ilu.factor)
        ilu.factor = ilu0(A, eltype(b))
        set_dim!(ilu, A, b)
    else
        ilu0!(ilu.factor, A)
    end
end

function update_preconditioner!(ilu::ILUZeroPreconditioner, A::StaticSparsityMatrixCSR, b, context::ParallelCSRContext, executor)
    if isnothing(ilu.factor)
        mb = A.minbatch
        max_t = max(size(A, 1) ÷ mb, 1)
        nt = min(nthreads(context), max_t)
        if nt == 1
            @debug "Setting up serial ILU(0)-CSR"
            F = ilu0_csr(A)
        else
            @debug "Setting up parallel ILU(0)-CSR with $(nthreads(td)) threads"
            part = context.partitioner
            lookup = generate_lookup(part, A, nt)
            F = ilu0_csr(A, lookup)
        end
        ilu.factor = F
        set_dim!(ilu, A, b)
    else
        ilu0_csr!(ilu.factor, A)
    end
end

function apply!(x, ilu::ILUZeroPreconditioner, y, arg...)
    factor = get_factorization(ilu)
    ilu_apply!(x, factor, y, arg...)
end

function ilu_f(type::Symbol)
    # Why must this be qualified?
    if type == :left
        f = forward_substitution!
    elseif type == :right
        f = backward_substitution!
    else
        f = ldiv!
    end
end

function ilu_apply!(x::AbstractArray{F}, f::AbstractILUFactorization, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
    T = eltype(f)
    if T == Float64
        ldiv!(x, f, y)
    else
        N = size(T, 1)
        T = eltype(T)
        Vt = SVector{N, T}
        as_svec = (x) -> unsafe_reinterpret(Vt, x, length(x) ÷ N)
        ldiv!(as_svec(x), f, as_svec(y))
    end
    return x
end

function ilu_apply!(x, f::AbstractILUFactorization, y, type::Symbol = :both)
    ldiv!(x, f, y)
end

function ilu_apply!(x::AbstractArray{F}, f::ILU0Precon{F}, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
    f! = ilu_f(type)
    f!(x, f, y)
end

# function ilu_apply!(x::AbstractArray{F}, f::CuSparseMatrix{F}, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
#     x .= y
#     ix = 'O'
#     sv2!('N', 'L', 'N', 1.0, f, x, ix)
#     sv2!('N', 'U', 'U', 1.0, f, x, ix)
# end

function ilu_apply!(x, ilu::ILU0Precon, y, type::Symbol = :both)
    T = eltype(ilu.l_nzval)
    N = size(T, 1)
    T = eltype(T)
    Vt = SVector{N, T}
    as_svec = (x) -> unsafe_reinterpret(Vt, x, length(x) ÷ N)

    # Solve by reinterpreting vectors to block (=SVector) vectors
    f! = ilu_f(type)
    f!(as_svec(x), ilu, as_svec(y))
    return x
end

function operator_nrows(ilu::ILUZeroPreconditioner)
    return ilu.dim[1]
end
