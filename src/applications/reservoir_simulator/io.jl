export preprocess_relperm_table

function preprocess_relperm_table(swof, ϵ = 1e-16)
    sw = swof[:, 1]
    krw = swof[:, 2]'
    # Change so table to be with respect to so,
    # and to be increasing with respect to input
    so = 1 .- sw
    so = so[end:-1:1]
    kro = swof[end:-1:1, 3]'
    # Subtract a tiny bit from the saturations at endpoints.
    # This is to ensure that the derivative ends up as zero
    # when evaluated at s corresponding to kr_max
    ensure_endpoints!(so, kro, ϵ)
    ensure_endpoints!(sw, krw, ϵ)
    s = [sw, so]
    krt = vcat(krw, kro)
    return s, krt
end

function ensure_endpoints!(x, f, ϵ)
    n = length(x)
    for i in (n-1):-1:2
        if f[i] != f[i-1]
            x[i] -= ϵ
            break
        end
    end
    for i in 1:(n-1)
        if f[i] != f[i+1]
            x[i] -= ϵ
            break
        end
    end
end