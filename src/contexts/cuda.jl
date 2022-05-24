
# CUDA context - everything on the single CUDA device attached to machine
struct SingleCUDAContext <: GPUJutulContext
    float_t::Type
    index_t::Type
    block_size
    device
    matrix_layout
    function SingleCUDAContext(float_t::Type = Float32, index_t::Type = Int64, block_size = 256, layout = EquationMajorLayout())
        @assert CUDA.functional() "CUDA must be functional for this context."
        return new(float_t, index_t, block_size, CUDADevice(), layout)
    end
end
matrix_layout(c::SingleCUDAContext) = c.matrix_layout
