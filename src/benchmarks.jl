export bench_half_face_flux, bench_half_face_flux_kernel

function bench_half_face_flux(name)
    # G = read_sim_graph(string("testgrids/", name, ".mat"))
    if isa(name, String)
        G = get_minimal_tpfa_grid_from_mrst(name)
    else
        G = name
    end
    nc = number_of_cells(G)
    nf = number_of_faces(G)
    cv = rand(nc)
    eq = rand(nc)
    fv = rand(nf)
    flux = similar(cv, 2*nf)
    println("Float64-version:")
    @time half_face_flux!(flux, cv, cv, G)
    
    tmp = ForwardDiff.Dual(1.0, 1.0)
    cvAD = Vector{typeof(tmp)}(undef, nc)
    for i in eachindex(cvAD)
        cvAD[i] = ForwardDiff.Dual(cv[i], 1.0)
    end
    println("AD-version:")
    fluxAD = similar(cvAD, 2*nf)
    @time half_face_flux!(fluxAD, cvAD, cvAD, G)
end

function bench_half_face_flux_kernel(name::String, doGPU = CUDA.functional())
    # G = read_sim_graph(string("testgrids/", name, ".mat"))
    G = read_sim_graph(name)
    nc = G.ncells
    nf = G.nfaces
    cv = rand(nc)
    eq = rand(nc)
    fv = rand(nf)
    flux = similar(cv, 2*nf)
    println("CPU raw function loop")
    @time half_face_flux!(flux, cv, cv, G)

    cpu_bz = 16
    kernel = half_face_flux_kernel(CPU(), cpu_bz)

    m = 2*nf
    self = G.self
    cells = G.cells
    hf = G.HalfFaceData
    println("CPU kernel, Float64")
    @time begin
        event = kernel(flux, cv, cv, hf, ndrange=m)
        wait(event)
    end

    tmp = ForwardDiff.Dual(1.0, 1.0)
    cvAD = Vector{typeof(tmp)}(undef, nc)
    cvAD2 = Vector{typeof(tmp)}(undef, nc)
    for i in eachindex(cvAD)
        cvAD[i] = ForwardDiff.Dual(cv[i], 1.0)
        cvAD2[i] = ForwardDiff.Dual(cv[i], 1.0)
    end
    fluxAD = similar(cvAD, 2*nf)
    println("CPU kernel, AD (Dual)")
    @time begin
        event = kernel(fluxAD, cvAD, cvAD2, hf, ndrange=m)
        wait(event)
    end
    if doGPU
        gpu_bz = 256
        kernel_gpu = half_face_flux_kernel(CUDADevice(), gpu_bz)


        cu_hf = CuArray(hf)
        cu_flux = CuArray(flux)
        cu_cv = CuArray(cv)
        cu_cv2 = CuArray(cv)

        println("GPU kernel, float")
        @time begin
            event = kernel_gpu(cu_flux, cu_cv, cu_cv, cu_hf, ndrange=m)
            wait(event)
        end

        cu_fluxAD = CuArray(fluxAD)
        cu_cvAD = CuArray(cvAD)
        cu_cvAD2 = CuArray(cvAD2)

        println("GPU kernel, AD")
        @time begin
            event = kernel_gpu(cu_fluxAD, cu_cvAD, cu_cvAD2, cu_hf, ndrange=m)
            wait(event)
        end
    end
end
