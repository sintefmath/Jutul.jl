using Test, StaticArrays, Jutul, SparseArrays
using Jutul: SparsePattern, LinearizedSystem, LinearizedBlock, MultiLinearizedSystem, BlockMajorLayout, EquationMajorLayout, DefaultContext
using Jutul: linear_operator, block_major_to_equation_major_view, equation_major_to_block_major_view

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

function system_1()
    a = @SMatrix [1.0 2; 3 4]
    b = @SMatrix [5.0 6; 7 8]
    c = @SMatrix [9.0 10; 11 12]
    
    x1 = @SVector [π, 3.0]
    x2 = @SVector [sqrt(3), 23.1]
    
    
    A_b = sparse([1, 1, 2], [1, 2, 1], [a, b, c], 2, 2)
    B = sparse([1.0 3 7.0; 10 6 1; 9 11 2; 7 13 5])
    C = sparse([0.1 3 7 2; 6 4 1.2 5; 1 1 6 3])
    D = sparse([9 8 7; 6 5 4; 3 2 1])

    X_b = [x1, x2]
    Y = [3.0, 7.2, 5.1]

    return (A = A_b, B = B, C = C, D = D, X = X_b, Y = Y)
end

function system_rand(;bz = 2, nb = 5, ns = 8, dens = 0.8)
    Mat = SMatrix{bz, bz}
    Vec = SVector{bz}

    A_b = sprand(Mat, nb, nb, dens)

    B = sprand(nb*bz, ns, dens)
    C = sprand(ns, nb*bz, dens)
    D = sprand(ns, ns, dens)

    X_b = rand(Vec, nb)
    Y = rand(ns)

    return (A = A_b, B = B, C = C, D = D, X = X_b, Y = Y)
end

function test_system(system)
    A_b, B, C, D, X_b, Y = system
    bz = length(eltype(X_b))
    A_s = block_system_to_scalar(A_b, reorder = true)
    # Vector to multiply with
    X_s = block_vector_to_scalar(X_b, reorder = true)
    # Scalar version of system
    
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
    # Test that the linear operator of a block system, acting on a scalar system
    # with that same ordering, produces the same result as the non-block ordering
    @test A_op*X_sb ≈ equation_major_to_block_major_view(A_s*X_s, bz)

    systems = [A_sys B_sys; C_sys D_sys]
    sys = MultiLinearizedSystem(systems, DefaultContext(), EquationMajorLayout())
    op = linear_operator(sys)
    V_op = [X_sb; Y]
    res_b = op*V_op
    res_s = M_s*V_s
    
    nf = length(X_s)
    # Test that the equation ordered part is ok
    @test res_b[nf+1:end] ≈ res_s[nf+1:end]
    # Test that the block ordered part is ok
    renum = equation_major_to_block_major_view(res_s[1:nf], bz)
    @test res_b[1:nf] ≈ renum    
end


@testset "Mixed block-scalar system" begin
    @testset "Test system 1" begin
        sys = system_1()
        test_system(sys)
    end
    block_sizes = 2:10
    matrix_sizes = 1:10
    for bz in block_sizes
        for ns in matrix_sizes
            for nb in matrix_sizes
                @testset "Rand" begin
                    sys = system_rand(bz = bz, ns = ns, nb = nb)
                    test_system(sys)
                end
            end
        end
    end
end
