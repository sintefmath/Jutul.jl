import Base.getindex

function Base.getindex(ls::MultiLinearizedSystem, i, j = i)
    return ls.subsystems[i, j]
end

function vector_residual(sys::Matrix{LinearizedSystem})
    r = map(vector_residual, diag(sys))
    return vcat(r...)
end
