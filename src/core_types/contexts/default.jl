
"Default context"
struct DefaultContext{Tv, Ti} <: CPUJutulContext{Tv, Ti}
    matrix_layout
    minbatch::Int64
    nthreads::Int64
    function DefaultContext(;
            matrix_layout = EquationMajorLayout(),
            minbatch = minbatch(nothing),
            nthreads = Threads.nthreads(),
            float_type = Float64,
            index_type = Int64
        )
        new{float_type, index_type}(matrix_layout, minbatch, nthreads)
    end
end

minbatch(c::DefaultContext) = c.minbatch
nthreads(c::DefaultContext) = c.nthreads
matrix_layout(c::DefaultContext) = c.matrix_layout
