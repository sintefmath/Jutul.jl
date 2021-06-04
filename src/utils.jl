export convert_to_immutable_storage

function convert_to_immutable_storage(dct::AbstractDict)
    for (key, value) in dct
        dct[key] = convert_to_immutable_storage(value)
    end
    return (; (Symbol(key) => value for (key, value) in dct)...)
end

function convert_to_immutable_storage(v::Any)
    # Silently do nothing
    return v
end


"Apply a function to each element in the fastest possible manner."
function fapply!(out, f, inputs...)
    # Example:
    # x, y, z equal length
    # then fapply!(z, *, x, y) is equal to a parallel call of
    # z .= x.*y
    # If JuliaLang Issue #19777 gets resolved we can get rid of fapply!
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = f(map((x) -> x[i], inputs)...)
    end
end

function fapply!(out::CuArray, f, inputs...)
    # Specialize fapply for GPU to get automatic kernel computation
    @. out = f(inputs...)
end

function get_matrix_view(v::AbstractVector, n, m, transp = false, offset = 0)
    r_l = view(v, (offset+1):(offset + n*m))
    if transp
        v = reshape(r_l, m, n)'
    else
        v = reshape(r_l, n, m)
    end
    return v
end

function get_row_view(v::AbstractVector, n, m, row, transp = false, offset = 0)
    v = get_matrix_view(v, n, m, transp, offset)
    view(v, row, :)
end
