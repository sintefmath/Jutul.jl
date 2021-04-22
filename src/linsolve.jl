export LinearizedSystem, solve!

struct LinearizedSystem
    jac
    r
    dx
end

function solve!(sys::LinearizedSystem)
    sys.dx .= -sys.jac\sys.r
end