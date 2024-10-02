using DrWatson
@quickactivate "SLIC"

include(srcdir("smooth.jl"))

using Plots, Random 

rng = Random.MersenneTwister(1234)

x = -1:0.01:1 
y = @. (x - x^2 + 3*x^3)*sin(10*x)
yn = y + 0.5*randn(rng, size(y))
ysm = smooth_ode(yn)

scatter(x,yn, label="Noisy", color=:black, alpha=0.7)
plot!(x,y, label="Clean", color=:dodgerblue, lw=6)
plot!(x, ysm, label="Denoised", color=:red, lw=4, ls=:dash)
xlabel!("x")
ylabel!("y")