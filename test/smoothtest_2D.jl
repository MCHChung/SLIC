using DrWatson
@quickactivate "SLIC"

include(srcdir("smooth.jl"))

using Plots, Random 

rng = Random.MersenneTwister(1234)

# create grid 
x = -1:0.01:1 
y = -2:0.01:2

# holds 2d values of a unnormalized gaussian 
u = zeros(length(x), length(y))
for i=1:length(x)
    for j=1:length(y)
        u[i,j] = @. exp(-x[i]^2-y[j]^2)
    end
end

# add noise and then try to denoise 
un = u .+ 0.1*randn(rng, size(u))
usm = smooth_pde(un)

# plot 
clims = (min(u...), max(u...))
p1 = surface(u, clims=clims)
title!("Clean")
p2 = surface(un, clims=clims)
title!("Noisy")
p3 = surface(usm, clims=clims)
title!("Denoised")
plot(p1,p2,p3, layout=(3,1), size=(400,900))