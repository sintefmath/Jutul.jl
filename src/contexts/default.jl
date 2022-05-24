
"Default context"
struct DefaultContext <: CPUJutulContext
    matrix_layout
    minbatch::Int64
    function DefaultContext(; matrix_layout = EquationMajorLayout(), minbatch = thread_batch(nothing))
        new(matrix_layout, minbatch)
    end
end

thread_batch(c::DefaultContext) = c.minbatch

matrix_layout(c::DefaultContext) = c.matrix_layout