
"Default context"
struct DefaultContext <: CPUJutulContext
    matrix_layout
    minbatch::Int64
    function DefaultContext(; matrix_layout = EquationMajorLayout(), minbatch = minbatch(nothing))
        new(matrix_layout, minbatch)
    end
end

minbatch(c::DefaultContext) = c.minbatch

matrix_layout(c::DefaultContext) = c.matrix_layout