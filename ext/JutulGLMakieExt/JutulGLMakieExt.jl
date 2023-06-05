module JutulGLMakieExt

using Jutul, GLMakie
    include("variables.jl")

    function Jutul.independent_figure(fig::Figure)
        display(GLMakie.Screen(), fig)
    end
end
