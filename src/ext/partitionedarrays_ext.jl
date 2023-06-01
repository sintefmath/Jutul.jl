export PArrayBackend, MPI_PArrayBackend

function simulate_parray
    
end

function parray_preconditioner_apply!

end

function parray_update_preconditioners!

end

function partition_distributed

end

function parray_linear_system_operator

end

abstract type PArrayBackend <: JutulBackend end

struct DebugPArrayBackend <: PArrayBackend end

struct JuliaPArrayBackend <: PArrayBackend end

struct MPI_PArrayBackend <: PArrayBackend end


struct PArraySimulator{T} <: Jutul.JutulSimulator
    backend::T
    storage
end

const MPISimulator = PArraySimulator{MPI_PArrayBackend}

struct PArrayExecutor{T} <: Jutul.JutulExecutor
    data
    mode::T
    rank::Int
    to_global::Vector{Int}
end

function PArrayExecutor(mode, rank, to_global;kwarg...)
    data = Dict{Symbol, Any}()
    for (k, v) in kwarg
        data[k] = v
    end
    return PArrayExecutor(data, mode, rank, to_global)
end
