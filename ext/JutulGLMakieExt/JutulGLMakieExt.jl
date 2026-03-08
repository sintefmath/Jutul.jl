module JutulGLMakieExt
    using Jutul, GLMakie
    include("variables.jl")
    include("explorer_3d.jl")

    function Jutul.independent_figure(fig::Figure)
        if get(ENV, "CI", "false") == "false"
            display(GLMakie.Screen(), fig)
        end
    end
end
