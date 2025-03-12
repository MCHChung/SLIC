using Plots , Measures

# this helps visualize the performance of the sparse regression algorithm 
# y axis are the model terms  i.e. 1, x, x^2, etc..
# x axis are the variables observed 
# blank entries are 0's i.e. the term is not included in the model output 

default(dpi=300, grid=false, fontfamily="computer modern")

function VisResults(Ξtrue, Ξes; clims=nothing)
    clims = isequal(clims, nothing) ? (min(Ξes...), max(Ξes...)) : clims
    p1 = heatmap(replace(Ξtrue, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="True/Exp. Model")
    p2 = heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="Est. Model")
    plot(p1,p2,layout=(1,2), size=(800,800), margin=7*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end

function VisResults(Ξtrue::Vector, Ξes::Vector; clims=nothing)
    Ξtrue = reshape(Ξtrue, (length(Ξtrue), 1))
    Ξes = reshape(Ξes, (length(Ξes), 1))

    clims = isequal(clims, nothing) ? (min(Ξes...), max(Ξes...)) : clims
    p1 = heatmap(replace(Ξtrue, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="True/Exp. Model")
    p2 = heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="Est. Model")
    plot(p1,p2,layout=(1,2), size=(800,800), margin=7*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end

function VisResults(Ξtrue, Ξes, label::String; clims=nothing)
    clims = isequal(clims, nothing) ? (min(Ξes...), max(Ξes...)) : clims
    p1 = heatmap(replace(Ξtrue, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, 
    clims=clims, title="True/Exp. Model")
    p2 = heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, 
    clims=clims, title="Est. Model : "*label)
    plot(p1,p2,layout=(1,2), size=(800,800), margin=7*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end

function VisResults(Ξes, label::String; clims=nothing)
    clims = isequal(clims, nothing) ? (min(Ξes...), max(Ξes...)) : clims
    heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, 
    axis=false, title="Est. Model : "*label, margins=7*Measures.mm, clims=clims)
    xlabel!("Variables")
    ylabel!("Model terms")
end

function VisResults(Ξtrue, Ξaic, Ξslic)
    clims = (min(Ξtrue...), max(Ξtrue...))
    p1 = heatmap(replace(Ξtrue, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="True/Exp. Model")
    p2 = heatmap(replace(Ξaic, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="AICc")
    p3 = heatmap(replace(Ξslic, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="SLIC")

    plot(p1,p2,p3,layout=(1,3), size=(1500,800), margin=5*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end