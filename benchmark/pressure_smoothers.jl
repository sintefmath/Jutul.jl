# Replay a saved pressure CSR/RHS without depending on JutulDarcy.
# Input: serialize(path, (; transposed_matrix=SparseMatrixCSC(...), rhs=...)).
# Run with --threads=10; repeat with --threads=1 for a fully serial baseline.
using Jutul, Jutul.KAPreconditioners, LinearAlgebra, SparseArrays
using Serialization, Statistics, TOML

function pressure_sample(f; repeats = 7)
    f()
    samples = [@timed f() for _ in 1:repeats]
    return Dict(
        "seconds" => median(s.time for s in samples),
        "bytes" => minimum(s.bytes for s in samples),
        "compile_seconds" => sum(s.compile_time for s in samples)
    )
end

function pressure_benchmark(input, output)
    BLAS.set_num_threads(1)
    data = deserialize(input)
    A = Jutul.StaticSparsityMatrixCSR(
        data.transposed_matrix;
        nthreads = Threads.nthreads(), minbatch = 8192, thread_type = :batch
    )
    b = copy(data.rhs)
    b ./= norm(b)
    policies = [
        ("serial_gs", GaussSeidel()),
        ("hybrid_1", HybridGaussSeidel(; npartitions = 1)),
        ("hybrid_contiguous", HybridGaussSeidel()),
        ("hybrid_metis", HybridGaussSeidel(; partitioner = :metis)),
        ("hybrid_32k", HybridGaussSeidel(; min_partition_size = 32768)),
    ]
    hierarchies = Any[]
    for (label, config) in policies
        println("Setup ", label); flush(stdout)
        options = AMGOptions(; smoother = config, coarse_size = 5)
        setup_amg(A, options) # Compile symbolic and numeric setup before timing.
        setup = @timed setup_amg(A, options)
        push!(hierarchies, (label, setup.value, setup.time))
    end
    results = Any[]
    for round in 1:3
        for (label, H, setup_seconds) in circshift(hierarchies, round - 1)
            x = similar(b)
            application = pressure_sample(() -> KAPreconditioners.apply!(x, H, b))
            numeric_update = pressure_sample(() -> resetup_amg!(H, A, :operators); repeats = 3)
            solver = Jutul.Krylov.GmresWorkspace(A, b; memory = 50)
            solves = Any[]
            for tolerance in (0.005, 1.0e-8)
                solve!() = Jutul.Krylov.gmres!(
                    solver, A, b;
                    N = H, rtol = tolerance, atol = 0.0, itmax = 200, history = true
                )
                timing = pressure_sample(solve!; repeats = 5)
                solved = solver.stats.solved
                relative_residual = norm(b - A * solver.x) / norm(b)
                @assert solved && relative_residual <= tolerance
                push!(
                    solves, merge(
                        timing, Dict(
                            "tolerance" => tolerance,
                            "iterations" => solver.stats.niter, "solved" => solved,
                            "true_relative_residual" => relative_residual
                        )
                    )
                )
            end
            levels = [
                Dict(
                    "rows" => size(level.A, 1), "nonzeros" => nnz(level.A),
                    "partitions" => level.smoother isa KAPreconditioners.HybridGaussSeidelState ?
                        length(level.smoother.schedule.offsets) - 1 : 1,
                    "interface_rows" => level.smoother isa KAPreconditioners.HybridGaussSeidelState ?
                        length(level.smoother.halo_rhs) : 0
                ) for level in H.levels
            ]
            row = Dict(
                "label" => label, "round" => round, "setup_seconds" => setup_seconds,
                "apply" => application, "numeric_update" => numeric_update,
                "solves" => solves, "levels" => levels
            )
            push!(results, row)
            println(label, " round=", round, " apply=", application, " solves=", solves); flush(stdout)
            open(output, "w") do io
                TOML.print(
                    io, Dict(
                        "input" => input, "threads" => Threads.nthreads(),
                        "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
                        "rows" => length(b), "nonzeros" => nnz(A), "results" => results
                    )
                )
            end
        end
    end
    return
end

if length(ARGS) != 2
    error("Usage: pressure_smoothers.jl INPUT.bin OUTPUT.toml")
end
pressure_benchmark(ARGS...)
