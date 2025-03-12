using DrWatson, Plots
@quickactivate "SLIC"
using MAT 

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern")

# load functions from src 
include(srcdir("sparse_regress.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("smooth.jl"))

# get data, originally from ==> https://www.nature.com/articles/s41467-019-12490-1 
data = load(datadir("exp_raw", "car.jld"))

# load variables
du = data["du"] ;
flag = data["flag"] # this is their way of identifying curve vs straightaway
v = data["v"]

# create time-delayed quantities
memory = 4
v_k1 = v[memory+1:end-1,:];
v_k2 = v[memory:end-2,:];
v_k3 = v[memory-1:end-3,:];
v_k4 = v[memory-2:end-4,:];

v = v[memory+2:end,:];
du = du[memory+2:end];
flag_ = flag[memory+2:end];

V = vcat(v', v_k1', v_k2', v_k3', v_k4')

# 2nd order library in V
function CarLib(V)
    n = size(V,1)
    θ = ones(size(V[1,:]))
    for i=1:n
        θ = hcat(θ, V[i,:])
    end

    for i=1:n
        for j=i:n
            θ = hcat(θ, V[i,:].*V[j,:])
        end
    end

    return θ
end

# remove outliers not included in the original paper 
badinds = findall(x -> x ∈[ -3979.0
5522.0
5747.0
5758.0
-6050.0
-6660.0], du)

goodinds = [i for i=1:length(du) if i ∉ badinds]

# plot 
p = scatter(du[goodinds], color=[flag_[goodinds][i]==0. ? :blue : :magenta for i=1:length(du[goodinds]) ], label= "Curve")
scatter!([0], [6000], color=:magenta, label="Straightaway")
ylims!(-5000, 5000)
xlabel!("Time step")
ylabel!("Differential motor input")
#wsave(plotsdir("autocar_vel_vs_du_hybrid.png"), p)

# parameters
tol = 0.5
trainpct = 60
num_batches = 500

# curve
inds = flag_[goodinds].== 0. 
θcar = CarLib(V[:, goodinds][:, inds])
du_in = reshape(du[goodinds][inds], (length(du[goodinds][inds]),1))

Ξaic0, _ , _= EnAdSR(θcar, du_in, "aic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξaicc0, _, _ = EnAdSR(θcar, du_in, "aicc", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξhqic0, _ , _= EnAdSR(θcar, du_in, "hqic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξbic0, _, _ = EnAdSR(θcar, du_in, "bic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξkic0, _ , _= EnAdSR(θcar, du_in, "kic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξbc0, _, _ = EnAdSR(θcar, du_in, "bc", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξslic0, _ , _= EnAdSR(θcar, du_in, "slic", tol=tol, num_batches = num_batches, trainpct=trainpct)

Ξtrue0 = [2660., -57.5, 48.] # true programmed control law for curve
Ξtrue0 = vcat(Ξtrue0, zeros(length(Ξslic0)-3))
Ξtrue0 = reshape(Ξtrue0, (length(Ξtrue0), 1))

# visualize 
# Curve
lΞtrue0 = log10.(abs.(Ξtrue0))
clims = (1,3.5)
pc_aic = VisResults(lΞtrue0, log10.(abs.(Ξaic0)), "AIC Curve", clims=clims)
pc_aicc = VisResults(lΞtrue0, log10.(abs.(Ξaicc0)), "AICc Curve", clims=clims)
pc_hqic = VisResults(lΞtrue0, log10.(abs.(Ξhqic0)), "HQIC Curve", clims=clims)
pc_bic = VisResults(lΞtrue0, log10.(abs.(Ξbic0)), "BIC Curve", clims=clims)
pc_kic = VisResults(lΞtrue0, log10.(abs.(Ξkic0)), "KIC Curve", clims=clims)
pc_bc = VisResults(lΞtrue0, log10.(abs.(Ξbc0)), "BC Curve", clims=clims)
pc_slic = VisResults(lΞtrue0, log10.(abs.(Ξslic0)), "SLIC Curve", clims=clims)

display(pc_aic)
display(pc_aicc)
display(pc_hqic)
display(pc_bic)
display(pc_kic)
display(pc_bc)
display(pc_slic)

#=
# save plots 
wsave(plotsdir("main", "aic_autocar_curve.png"), pc_aic)
wsave(plotsdir("main", "aicc_autocar_curve.png"), pc_aicc)
wsave(plotsdir("main", "hqic_autocar_curve.png"), pc_hqic)
wsave(plotsdir("main", "bic_autocar_curve.png"), pc_bic)
wsave(plotsdir("main", "kic_autocar_curve.png"), pc_kic)
wsave(plotsdir("main", "bc_autocar_curve.png"), pc_bc)
wsave(plotsdir("main", "slic_autocar_curve.png"), pc_slic)
=# 

# repeat for straightaway
inds = flag_[goodinds].== 1. 
θcar = CarLib(V[:, goodinds][:, inds])
du_in = reshape(du[goodinds][inds], (length(du[goodinds][inds]),1))

Ξaic1, _ , _= EnAdSR(θcar, du_in, "aic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξaicc1, _, _ = EnAdSR(θcar, du_in, "aicc", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξhqic1, _ , _= EnAdSR(θcar, du_in, "hqic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξbic1, _, _ = EnAdSR(θcar, du_in, "bic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξkic1, _ , _= EnAdSR(θcar, du_in, "kic", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξbc1, _, _ = EnAdSR(θcar, du_in, "bc", tol=tol, num_batches = num_batches, trainpct=trainpct)
Ξslic1, _ , _= EnAdSR(θcar, du_in, "slic", tol=tol, num_batches = num_batches, trainpct=trainpct)

Ξtrue1 = [3610., -57.5, 48.] # true programmed control law for straightaway
Ξtrue1 = vcat(Ξtrue1, zeros(length(Ξslic1)-3))
Ξtrue1 = reshape(Ξtrue1, (length(Ξtrue1), 1))

# visualize
# Straightaway
lΞtrue1 = log10.(abs.(Ξtrue1))
clims = (1,3.5)
ps_aic = VisResults(lΞtrue1, log10.(abs.(Ξaic1)), "AIC Straightaway", clims=clims)
ps_aicc = VisResults(lΞtrue1, log10.(abs.(Ξaicc1)), "AICc Straightaway", clims=clims)
ps_hqic = VisResults(lΞtrue1, log10.(abs.(Ξhqic1)), "HQIC Straightaway", clims=clims)
ps_bic = VisResults(lΞtrue1, log10.(abs.(Ξbic1)), "BIC Straightaway", clims=clims)
ps_kic = VisResults(lΞtrue1, log10.(abs.(Ξkic1)), "KIC Straightaway", clims=clims)
ps_bc = VisResults(lΞtrue1, log10.(abs.(Ξbc1)), "BC Straightaway", clims=clims)
ps_slic = VisResults(lΞtrue1, log10.(abs.(Ξslic1)), "SLIC Straightaway", clims=clims)

display(ps_aic)
display(ps_aicc)
display(ps_hqic)
display(ps_bic)
display(ps_kic)
display(ps_bc)
display(ps_slic)

#=
# save plots
wsave(plotsdir("main", "aic_autocar_straight.png"), ps_aic)
wsave(plotsdir("main", "aicc_autocar_straight.png"), ps_aicc)
wsave(plotsdir("main", "hqic_autocar_straight.png"), ps_hqic)
wsave(plotsdir("main", "bic_autocar_straight.png"), ps_bic)
wsave(plotsdir("main", "kic_autocar_straight.png"), ps_kic)
wsave(plotsdir("main", "bc_autocar_straight.png"), ps_bc)
wsave(plotsdir("main", "slic_autocar_straight.png"), ps_slic)
=# 


# plot errors 
rel_err(Ξtrue, Ξes) = sum(abs, Ξtrue - Ξes)/sum(abs, Ξtrue)
Ξs0 = [Ξslic0, Ξaic0, Ξaicc0, Ξhqic0, Ξbic0, Ξkic0, Ξbc0]
Ξs1 = [Ξslic1, Ξaic1, Ξaicc1, Ξhqic1, Ξbic1, Ξkic1, Ξbc1]
marker_shape = [:circle , :utriangle, :dtriangle, :square, :star, :hexagon, :pentagon]
marker_color = [:red, :black, :blue, :green, :darkorange, :gold, :purple]
errs = []
pc = plot()
ps = plot()
for i=1:length(Ξs0)
    err0 = rel_err(Ξtrue0, Ξs0[i])
    err1 = rel_err(Ξtrue1, Ξs1[i])
    err = [err0, err1]
    push!(errs, [err0, err1])
    scatter!(pc, [i], [err[1]], ms=10, color=marker_color[i], marker=marker_shape[i], label=false)
    ylims!(0.0,0.041)
    title!(pc, "Rel Err: Curve ")
    scatter!(ps, [i], [err[2]], ms=10, color=marker_color[i],  marker=marker_shape[i], label=false)
    ylims!(0.0098,0.028)
    title!(ps, "Rel Err: Straight ")
end

p = plot(pc, ps, figure=(1,2), size=(1500,400), margins=7*Measures.mm, xtickfontsize=16, ytickfontsize=16)
display(p)
#wsave(plotsdir("main", "car_err.png"), p)
