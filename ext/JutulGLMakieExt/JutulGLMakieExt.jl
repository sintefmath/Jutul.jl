module JutulGLMakieExt
using Jutul, GLMakie

function Jutul.independent_figure(fig::Figure)
    if Makie.current_backend() === GLMakie
        if get(ENV, "CI", "false") == "false"
            display(GLMakie.Screen(), fig)
        end
    else
        display(fig)
    end
    return fig
end
end
