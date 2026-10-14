module JutulGLMakieExt
using Jutul, GLMakie

Jutul.allocate_plot_buffer(values, ::Type{<:GLMakie.Screen}) = GLMakie.Buffer(values)

function Jutul.update_plot_buffer!(buffer::GLMakie.Buffer, values)
    buffer[1:length(values)] = values
    return buffer
end

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
