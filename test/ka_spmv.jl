using Jutul
using CUDA
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test
import Adapt

function spmv_capacity_buffer(buffer, backend)
    allocation = similar(buffer, length(buffer) + 2)
    copyto!(allocation, 1, buffer, 1, length(buffer))
    return Jutul.KAPreconditioners.buffer_prefix(allocation, length(buffer))
end

function test_vendor_spmv(backend)
    @testset "Scalar SpMV $T $Ti vendor=$use_vendor_linalg" for
            T in (Float32, Float64, ComplexF32, ComplexF64),
            Ti in (Int32, Int64), use_vendor_linalg in (true, false)
        context = KernelAbstractionsContext(
            backend; float_type = real(T), index_type = Ti, use_vendor_linalg
        )
        host = sparse([1, 1, 3, 4], [1, 3, 2, 3], T[2, -1, 3, 4], 4, 3)
        matrix = Adapt.adapt(context, Jutul.StaticSparsityMatrixCSR(copy(transpose(host))))
        x = Adapt.adapt(backend, T[1, 2, 3])
        y = Adapt.adapt(backend, fill(T(NaN), 4))
        @test mul!(y, matrix, x) === y
        @test Array(y) ≈ host * Array(x)
        plan = matrix.vendor_linalg[]
        @test isnothing(plan) == !use_vendor_linalg
        if use_vendor_linalg
            @test plan.matrix.nzVal === matrix.nzval
        end
        # Rebind the cached descriptors to different vectors and dense views.
        other_x = Adapt.adapt(backend, T[9, 3, 2, 1, 9])
        other_y = Adapt.adapt(backend, T[9, 4, 3, 2, 1, 9])
        xv, yv = view(other_x, 2:4), view(other_y, 2:5)
        expected = T(2) * host * Array(xv) + T(3) * Array(yv)
        mul!(yv, matrix, xv, 2.0, 3.0)
        @test Array(yv) ≈ expected
        @test matrix.vendor_linalg[] === plan
        fill!(y, T(NaN))
        mul!(y, matrix, x, 1.0, 0.0)
        @test Array(y) ≈ host * Array(x)
        # Assembly changes only values; vendor storage must observe them.
        matrix.nzval .*= T(2)
        mul!(y, matrix, x)
        @test Array(y) ≈ T(2) * host * Array(x)
        @test matrix.vendor_linalg[] === plan
        @test_throws DimensionMismatch mul!(view(y, 1:3), matrix, x)
        @test_throws DimensionMismatch mul!(y, matrix, view(x, 1:2), 1, 0)
    end

    @testset "Block SpMV $T $Ti vendor=$use_vendor_linalg" for
            T in (Float32, Float64, ComplexF32, ComplexF64),
            Ti in (Int32, Int64), use_vendor_linalg in (true, false)
        blocks = [SMatrix{2, 2, T}(1, 2, 3, 4),
                  SMatrix{2, 2, T}(2, -1, 4, 3),
                  SMatrix{2, 2, T}(3, 1, -2, 2)]
        host = Jutul.StaticSparsityMatrixCSR(
            blocks, Ti[1, 3, 2], Ti[1, 3, 4], 2, 3, CPU()
        )
        context = KernelAbstractionsContext(
            backend; float_type = real(T), index_type = Ti, use_vendor_linalg
        )
        matrix = Adapt.adapt(context, host)
        host_x = [SVector{2, T}(1, 2), SVector{2, T}(3, 4), SVector{2, T}(5, 6)]
        expected = [blocks[1] * host_x[1] + blocks[2] * host_x[3], blocks[3] * host_x[2]]
        x = Adapt.adapt(backend, host_x)
        y = Adapt.adapt(backend, fill(SVector{2, T}(NaN, NaN), 2))
        mul!(y, matrix, x)
        @test Array(y) ≈ expected
        plan = matrix.vendor_linalg[]
        @test isnothing(plan) == !use_vendor_linalg
        other_x, other_y = copy(x), copy(y)
        mul!(other_y, matrix, other_x, 2.0, 3.0)
        @test Array(other_y) ≈ 5 .* expected
        matrix.nzval .*= T(2)
        fill!(y, SVector{2, T}(NaN, NaN))
        mul!(y, matrix, x, 1.0, 0.0)
        @test Array(y) ≈ 2 .* expected
        @test matrix.vendor_linalg[] === plan
    end

    @testset "SpMV with reusable capacity buffers" for use_vendor_linalg in (true, false)
        for block in (false, true)
            values = block ? [SMatrix{2, 2}(1.0, 2.0, 3.0, 4.0)] : [2.0]
            host = Jutul.StaticSparsityMatrixCSR(values, Int64[1], Int64[1, 2], 1, 1, CPU())
            source = Adapt.adapt(KernelAbstractionsContext(backend; use_vendor_linalg), host)
            matrix = Jutul.StaticSparsityMatrixCSR(
                spmv_capacity_buffer(source.nzval, backend),
                spmv_capacity_buffer(source.colval, backend),
                spmv_capacity_buffer(source.rowptr, backend), 1, 1, backend;
                use_vendor_linalg
            )
            host_x = block ? [SVector(1.0, 2.0)] : [3.0]
            expected = [values[1] * host_x[1]]
            x = spmv_capacity_buffer(Adapt.adapt(backend, host_x), backend)
            y = spmv_capacity_buffer(Adapt.adapt(backend, zero.(expected)), backend)
            mul!(y, matrix, x)
            @test Array(y) ≈ expected
            plan = matrix.vendor_linalg[]
            @test isnothing(plan) == !use_vendor_linalg
            mul!(y, matrix, x, 2.0, 3.0)
            @test Array(y) ≈ 5 .* expected
            @test matrix.vendor_linalg[] === plan
        end
    end
end

if CUDA.functional()
    @testset "CUDA cached vendor SpMV" begin
        test_vendor_spmv(CUDA.CUDABackend())
    end
end

# Run the same checks when AMDGPU has been loaded by the test environment.
if isdefined(Main, :AMDGPU) && AMDGPU.functional()
    @testset "AMDGPU cached vendor SpMV" begin
        test_vendor_spmv(AMDGPU.ROCBackend())
    end
end
