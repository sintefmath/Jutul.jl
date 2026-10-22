"Default CPU context. See [`secondary_scheme`](@ref) for secondary-variable scheduling."
struct DefaultContext <: CPUJutulContext
    matrix_layout
    minbatch::Int64
    nthreads::Int64
    secondary_scheme::Symbol
    function DefaultContext(; matrix_layout = EquationMajorLayout(), minbatch = minbatch(nothing), nthreads = Threads.nthreads(), secondary_scheme = :flat)
        check_secondary_scheme(secondary_scheme)
        return new(matrix_layout, minbatch, nthreads, secondary_scheme)
    end
end

minbatch(c::DefaultContext) = c.minbatch
nthreads(c::DefaultContext) = c.nthreads
matrix_layout(c::DefaultContext) = c.matrix_layout
secondary_scheme(c::DefaultContext) = c.secondary_scheme

function Base.adjoint(c::DefaultContext)
    return DefaultContext(; matrix_layout = adjoint(matrix_layout(c)),
        minbatch = minbatch(c), nthreads = nthreads(c), secondary_scheme = secondary_scheme(c))
end
