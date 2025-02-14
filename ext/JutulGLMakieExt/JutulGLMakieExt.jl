module JutulGLMakieExt

using Jutul, GLMakie
    include("variables.jl")

    function Jutul.independent_figure(fig::Figure)
        if get(ENV, "CI", "false") == "false"
            display(GLMakie.Screen(), fig)
        end
    end
end
