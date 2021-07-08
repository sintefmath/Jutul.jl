import Base.getindex

function Base.getindex(ls::MultiLinearizedSystem, i, j = i)
    return ls.subsystems[i, j]
end
