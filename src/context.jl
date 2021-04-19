abstract type TervContext end
abstract type GPUTervContext end
abstract type CPUTervContext end


struct SingleCUDAContext <: GPUTervContext

end

struct SharedMemoryContext <: CPUTervContext

end