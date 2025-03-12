using DrWatson, Test
@quickactivate "SLIC"

# Here you include files using `srcdir`
include(srcdir("generate_ode_data.jl"))
include(srcdir("galerkin_proj.jl"))
include(srcdir("sparse_regress.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("smooth.jl"))

# sys::Int, tspan, dt, u0::Vector, ps::Vector
# gen Lorenz data 
sys = 1
tspan = (0., 10.)
dt = 1e-3
u0 = [-8., 7., 27]
ps = [10., 28., 8/3]
Xtrue, ts = GenData(sys, tspan, dt, u0, ps)

# build library
p = 10
#_,_,wind = FindW(X,ts,ws=21:2:121)
function LorLib(X,ts, wind; p=10)
    n = size(X,1)
    θ = wInt_sc(X[1,:], ts, wind, p=p)
    for i=2:n
        θ = hcat(θ,wInt_sc(X[i,:], ts, wind, p=p) )
    end

    for i=1:n
        for j=i:n
            θ = hcat(θ,wInt_sc(X[i,:].*X[j,:], ts, wind, p=p) )
        end
    end

    for i=1:n
        for j=i:n
            for k=j:n
                θ = hcat(θ,wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p) )
            end
        end
    end

    return θ
end

#display(plot(Xtrue[1,:], Xtrue[2,:], Xtrue[3,:]))

NoisePct = 10 
η = NoisePct*mean(std(Xtrue, dims=2))/100
Xn = Xtrue + η.*randn(size(Xtrue))
Xsm = smooth_ode(Xn)
_,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)

θ = LorLib(Xsm, ts, wind,p=p)
qt = dwInt_sc(Xsm, ts, wind, p=p)
ic = "slic"
Ξ, _ = EnAdSR(θ, qt', ic, tol=0.7, c=0, num_batches=10)
# get true Ξ and visualize
files = readdir(datadir("sims", "ode_results_main"))
lordata = load(datadir("sims", "ode_results_main", "lor_results_main.jld"))
Ξtrue =  lordata["Ξtrue"]

VisResults(Ξtrue, Ξ)
