using Test, StaticArrays, Terv, SparseArrays
using Terv: SparsePattern, LinearizedSystem, LinearizedBlock, MultiLinearizedSystem, BlockMajorLayout, EquationMajorLayout, DefaultContext
using Terv: linear_operator

function to_sparse_pattern(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    I, J, _ = findnz(A)
    n, m = size(A)
    layout = matrix_layout(A)
    block_n, block_m = block_dims(A)
    return SparsePattern(I, J, n, m, layout, block_n, block_m)
end

function matrix_layout(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:Real, Ti}
    return EquationMajorLayout()
end

function matrix_layout(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:StaticMatrix, Ti}
    layout = BlockMajorLayout()
    return layout
end

matrix_layout(A::AbstractVector{T}) where {T<:StaticVector} = BlockMajorLayout()
matrix_layout(A) = EquationMajorLayout()

function block_dims(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:Real, Ti}
    return (1, 1)
end

function block_dims(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:StaticMatrix, Ti}
    n, m = size(Tv)
    return (n, m)
end

block_dims(A::AbstractVector) = 1
block_dims(A::AbstractVector{T}) where T<:StaticVector = length(T)

function LinearizedSystem(A, r = nothing)
    pattern = to_sparse_pattern(A)
    layout = matrix_layout(A)
    context = DefaultContext(matrix_layout = layout)
    sys = LinearizedSystem(pattern, context, layout, r = r)
    J = sys.jac
    for (i, j, entry) in zip(findnz(A)...)
        J[i, j] = entry
    end
    return sys
end

function LinearizedBlock(A, bz::Tuple, row_layout, col_layout)
    pattern = to_sparse_pattern(A)
    layout = matrix_layout(A)
    context = DefaultContext(matrix_layout = layout)
    sys = LinearizedBlock(pattern, context, layout, row_layout, col_layout, bz)
    J = sys.jac
    for (i, j, entry) in zip(findnz(A)...)
        J[i, j] = entry
    end
    return sys
end
# Make the same system twice
# Convert to block system

function block_system_to_scalar(A; reorder = false)
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()

    n, m = size(A)
    bz = size(eltype(A), 1)
    # We know the sizes, speed things up by preallocating
    nz_count = nnz(A)
    sizehint!(I, nz_count*bz)
    sizehint!(J, nz_count*bz)
    sizehint!(V, nz_count*bz)

    for (i, j, entry) in zip(findnz(A)...)
        for k in 1:bz
            for l in 1:bz
                if reorder
                    newi = (k-1)*n + i
                    newj = (l-1)*m + j
                else
                    newi = (i-1)*bz + k
                    newj = (j-1)*bz + l
                end
                push!(I, newi)
                push!(J, newj)
                push!(V, entry[k, l])
            end
        end
    end
    return sparse(I, J, V, n*bz, m*bz)
end

function block_vector_to_scalar(x; reorder = false)
    n = length(x)
    bz = length(eltype(x))

    newx = zeros(bz*n)
    if reorder
        for i = 1:n
            for b = 1:bz
                newx[(b-1)*n + i] = x[i][b]
            end
        end
    else
        for i = 1:n
            for b = 1:bz
                newx[(i-1)*bz + b] = x[i][b]
            end
        end
    end
    return newx
end

a = @SMatrix [1.0 2; 3 4]
b = @SMatrix [5.0 6; 7 8]
c = @SMatrix [9.0 10; 11 12]

x1 = @SVector [π, 3.0]
x2 = @SVector [sqrt(3), 23.1]

bz = length(x1)

A_b = sparse([1, 1, 2], [1, 2, 1], [a, b, c], 2, 2)
A_s = block_system_to_scalar(A_b, reorder = true)
# Vector to multiply with
X_b = [x1, x2]
X_s = block_vector_to_scalar(X_b, reorder = true)
# Scalar version of system
B = sparse([1.0 3 7.0; 10 6 1; 9 11 2; 7 13 5])
C = sparse([0.1 3 7 2; 6 4 1.2 5; 1 1 6 3])
D = sparse([9 8 7; 6 5 4; 3 2 1])

Y = [3.0, 7.2, 5.1]

M_s = [A_s B; C D]
V_s = [X_s; Y]
## Multi with standard types
## Mult as linearized system
X_sb = block_vector_to_scalar(X_b, reorder = false)

A_sys = LinearizedSystem(A_b)
B_sys = LinearizedBlock(B, (bz, 1), BlockMajorLayout(), EquationMajorLayout())
C_sys = LinearizedBlock(C, (1, bz), EquationMajorLayout(), BlockMajorLayout())
D_sys = LinearizedSystem(D)


A_op = linear_operator(A_sys)
# @test A_op*X_sb
B_op = linear_operator(B_sys)
B_op*Y
##
systems = [A_sys B_sys; C_sys D_sys]
sys = MultiLinearizedSystem(systems, DefaultContext(), EquationMajorLayout())
op = linear_operator(sys)
V_op = [X_sb; Y]
res_b = op*V_op
res_s = M_s*V_s

nf = length(X_s)
@test res_b[nf+1:end] == res_s[nf+1:end]
##
renum = collect(reshape(reshape(res_s[1:nf], bz, :)', :))
@test res_b[1:nf] == renum
##
S = sys.subsystems
d = size(S, 2)
ops = map(linear_operator, S)

#
# ops = map(linear_operator, vec(permutedims(S)))

# op = hvcat(d, ops...)


op = vcat(hcat(ops[1, 1], ops[1, 2]), hcat(ops[2, 1], ops[2, 2]))
op*V_op
