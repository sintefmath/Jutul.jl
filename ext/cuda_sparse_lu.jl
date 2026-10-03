const RF = isdefined(CUDA, :cuSOLVER) ? CUDA.cuSOLVER : CUDA.CUSOLVER

mutable struct CUDARFFactor{V, I, H, R, C}
    handle::H
    rowptr::I
    colval::I
    values::V
    p::I
    q::I
    temporary::V
    solution::V
    n::Int
    pattern_rowptr::Vector{Int}
    pattern_colval::Vector{Int}
    source_rowptr::R
    source_colval::C
end

function rf_host_csr(A::StaticSparsityMatrixCSR)
    return Int32.(A.rowptr .- 1), Int32.(A.colval .- 1), Float64.(A.nzval)
end

function setup_cuda_sparse_lu(A::StaticSparsityMatrixCSR{Float64})
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sparse LU requires a square matrix"))
    host = KAPreconditioners.host_csr(A)
    sparse_host = KAPreconditioners.sparse_matrix(host)
    control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int64)
    control[SparseArrays.UMFPACK.JL_UMFPACK_SCALE] = 0.0
    initial = lu(sparse_host; control)
    L = KAPreconditioners.csr_matrix(sparse(initial.L); index_type = Int32)
    U = KAPreconditioners.csr_matrix(sparse(initial.U); index_type = Int32)
    ap, ai, ax = rf_host_csr(host)
    lp, li, lx = rf_host_csr(L)
    up, ui, ux = rf_host_csr(U)
    p = Int32.(initial.p .- 1)
    q = Int32.(initial.q .- 1)
    handle_ref = Ref{RF.cusolverRfHandle_t}()
    RF.cusolverRfCreate(handle_ref)
    handle = handle_ref[]
    try
        RF.cusolverRfSetMatrixFormat(
            handle, RF.CUSOLVERRF_MATRIX_FORMAT_CSR,
            RF.CUSOLVERRF_UNIT_DIAGONAL_STORED_L
        )
        RF.cusolverRfSetupHost(
            n, length(ax), ap, ai, ax,
            length(lx), lp, li, lx,
            length(ux), up, ui, ux, p, q, handle
        )
        RF.cusolverRfAnalyze(handle)
        RF.cusolverRfRefactor(handle)
        factor = CUDARFFactor(
            handle, CuArray(ap), CuArray(ai), CuArray(ax),
            CuArray(p), CuArray(q), CUDA.zeros(Float64, n), CUDA.zeros(Float64, n),
            n, Int.(host.rowptr), Int.(host.colval), A.rowptr, A.colval
        )
        finalizer(factor) do state
            RF.cusolverRfDestroy(state.handle)
        end
        return KAPreconditioners.SparseLU(factor)
    catch
        RF.cusolverRfDestroy(handle)
        rethrow()
    end
end

function KAPreconditioners.resetup_sparse_lu!(
        S::KAPreconditioners.SparseLU{<:CUDARFFactor},
        A::StaticSparsityMatrixCSR{Float64}
    )
    F = S.factorization
    KAPreconditioners.sparse_lu_same_pattern(S, A) ||
        throw(ArgumentError("sparse LU resetup requires the same CSR pattern"))
    # RF reuses the original pivot permutation. It can fail with a zero pivot,
    # or succeed with an unsuitable permutation after the values change. These
    # systems are small, so recompute pivoting for every resetup.
    S.factorization = setup_cuda_sparse_lu(A).factorization
    finalize(F)
    return S
end

function KAPreconditioners.setup_sparse_lu(
        matrix::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B,
        }
    ) where {
        Tv <: Union{Float32, Float64, ComplexF32, ComplexF64},
        Ti <: Integer, V, I, R, B <: CUDA.CUDABackend,
    }
    if applicable(KAPreconditioners.setup_preferred_sparse_lu, matrix)
        return KAPreconditioners.setup_preferred_sparse_lu(matrix)
    elseif Tv === Float64
        return setup_cuda_sparse_lu(matrix)
    else
        return invoke(
            KAPreconditioners.setup_sparse_lu,
            Tuple{StaticSparsityMatrixCSR}, matrix
        )
    end
end

function LinearAlgebra.ldiv!(
        x, S::KAPreconditioners.SparseLU{<:CUDARFFactor}, b
    )
    F = S.factorization
    length(x) == F.n && length(b) == F.n || throw(DimensionMismatch())
    copyto!(F.solution, b)
    RF.cusolverRfSolve(
        F.handle, F.p, F.q, 1, F.temporary, F.n, F.solution, F.n
    )
    copyto!(x, F.solution)
    return x
end

function Jutul.KernelExecution.factorize_linear_system(
        ::typeof(lu),
        matrix::StaticSparsityMatrixCSR{
            Float64, Ti, V, I, R, B,
        }
    ) where {Ti <: Integer, V, I, R, B <: CUDA.CUDABackend}
    return KAPreconditioners.setup_sparse_lu(matrix)
end
