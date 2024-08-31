abstract type NFVMDiscretization <: KGradDiscretization

end

struct NFVMLinearDiscretization{T} <: NFVMDiscretization
    left::Int
    right::Int
    T_left::T
    T_right::T
    mpfa::Vector{Tuple{Int, T}}
end

function Base.show(io::IO, ft::NFVMLinearDiscretization{T}) where T
    l, r = cell_pair(ft)
    print(io, "NFVMLinearDiscretization{$T} $(l)→$(r)")
    compact = get(io, :compact, false)
    if !compact
        avg = (abs(ft.T_left) + abs(ft.T_right))/2.0
        print(io, ", T≈$avg ($(length(ft.mpfa)) MPFA points)")
    end
end


struct NFVMNonLinearDiscretization{T} <: NFVMDiscretization
    ft_left::NFVMLinearDiscretization{T}
    ft_right::NFVMLinearDiscretization{T}
    scheme::Symbol
end

function NFVMNonLinearDiscretization(l, r; scheme = :ntpfa)
    @assert scheme in (:ntpfa, :nmpfa)
    return NFVMNonLinearDiscretization(l, r, scheme)
end

Jutul.cell_pair(x::NFVMNonLinearDiscretization) = Jutul.cell_pair(x.ft_left)
Jutul.cell_pair(x::NFVMLinearDiscretization) = (x.left, x.right)

function Base.show(io::IO, ft::NFVMNonLinearDiscretization{T}) where T
    l, r = cell_pair(ft)
    print(io, "NFVMNonLinearDiscretization{$T} $(l)→$(r)")
    compact = get(io, :compact, false)
    if !compact
        L = ft.ft_left
        R = ft.ft_right
        avg = (abs(L.T_left) + abs(L.T_right) + abs(R.T_left) + abs(R.T_right))/4.0
        print(io, ", T≈$avg ($(length(L.mpfa)) + $(length(R.mpfa)) MPFA points)")
    end
end

function merge_to_avgmpfa(a::NFVMLinearDiscretization{T}, b::NFVMLinearDiscretization{T}) where {T}
    function merge_in!(next, x)
        for el in x.mpfa
            c, v = el
            v /= 2.0
            if haskey(next, c)
                next[c] += v
            else
                next[c] = v
            end
        end
        return next
    end
    l, r = Jutul.cell_pair(a)
    @assert (l, r) == Jutul.cell_pair(b)
    next = Dict{Int, T}()
    merge_in!(next, a)
    merge_in!(next, b)
    mpfa = Vector{Tuple{Int, T}}()
    for (c, v) in pairs(next)
        push!(mpfa, (c, v))
    end
    T_l = (a.T_left + b.T_left)/2.0
    T_r = (a.T_right + b.T_right)/2.0
    return NFVMLinearDiscretization(l, r, T_l, T_r, mpfa)
end