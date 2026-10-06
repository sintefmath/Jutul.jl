# A fixed hierarchy is also a fixed launch sequence. Capture it once, rather
# than dispatching, converting arguments and submitting each kernel per apply.
struct CUDAAMGExecution{V, G, M, L, O}
    input::V
    output::V
    graph::G
    managed::M
    levels::L
    options::O
    block_size::Int
end

function cuda_amg_graph_eligible(H)
    H.options.smoother isa HybridGaussSeidel || return false
    for level in H.levels
        state = level.smoother
        state isa KAPreconditioners.HybridGaussSeidelState || return false
        # SpSV's internal analysis may change during numeric update. Retain
        # its regular launch path; these plans only read owned array storage.
        state.native isa Union{CUDAPartitionHybridGSFactor, CUDADiagonalHybridGSFactor} || return false
    end
    coarse = H.levels[end].coarse_solver
    return isnothing(coarse) || coarse isa KAPreconditioners.SparseLU{<:KAPreconditioners.KASparseLUFactor}
end

function cuda_amg_managed_buffers(H, input, output)
    adaptor = CUDA.KernelAdaptor()
    # Collect only the array fields consumed by the captured kernels, once.
    # Holding the managed buffers also keeps graph pointers alive, and lets
    # CUDA track ownership when the hierarchy moves to another task/stream.
    add(array::CuArray) = (Adapt.adapt(adaptor, array); nothing)
    add(other) = nothing
    add_fields(object) = isnothing(object) ? nothing : foreach(add, (getfield(object, f) for f in fieldnames(typeof(object))))
    add(input)
    add(output)
    for level in H.levels
        add_fields(level)
        add_fields(level.A)
        add_fields(level.P)
        add_fields(level.Pt)
        add_fields(level.smoother)
        native = level.smoother.native
        add_fields(native)
        native isa CUDAPartitionHybridGSFactor && add_fields(native.matrix)
    end
    coarse = H.levels[end].coarse_solver
    isnothing(coarse) || add_fields(coarse.factorization)
    return unique!(adaptor.managed)
end

function cuda_build_amg_execution(H, b)
    input, output = similar(b), similar(b)
    copyto!(input, b)
    cycle = () -> KAPreconditioners.vcycle!(output, input, H, 1;
        residual = input, zero_initial = true)
    # Compile and prepare every call before capture. Numeric resets write the
    # existing arrays in place, so they do not require recapture or graph update.
    cycle()
    graph = CUDA.instantiate(CUDA.capture(cycle))
    managed = cuda_amg_managed_buffers(H, input, output)
    return CUDAAMGExecution(input, output, graph, managed, H.levels, H.options, H.block_size)
end

function KAPreconditioners.apply_amg_execution!(x, H, b, ::CUDA.CUDABackend)
    H.execution = nothing
    return false
end

function KAPreconditioners.apply_amg_execution!(x::CuVector{T}, H::AMGHierarchy{T},
        b::CuVector{T}, ::CUDA.CUDABackend) where {T}
    CUDA.is_capturing() && return false
    execution = H.execution
    if !(execution isa CUDAAMGExecution) || execution.levels !== H.levels ||
            execution.options !== H.options || execution.block_size != H.block_size
        cuda_amg_graph_eligible(H) || return false
        execution = cuda_build_amg_execution(H, b)
        H.execution = execution
    end
    copyto!(execution.input, b)
    CUDA.with_managed(execution.managed) do
        CUDA.launch(execution.graph)
    end
    copyto!(x, execution.output)
    return true
end
