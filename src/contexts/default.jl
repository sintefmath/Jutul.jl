
"Default context"
struct DefaultContext <: CPUJutulContext
    matrix_layout
    minbatch::Int64
    nthreads::Int64
    function DefaultContext(; matrix_layout = EquationMajorLayout(), minbatch = minbatch(nothing), nthreads = Threads.nthreads())
        new(matrix_layout, minbatch, nthreads)
    end
end

minbatch(c::DefaultContext) = c.minbatch
nthreads(c::DefaultContext) = c.nthreads
matrix_layout(c::DefaultContext) = c.matrix_layout
