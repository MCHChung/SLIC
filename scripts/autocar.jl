using DrWatson
@quickactivate "SLIC"
using MAT 

# load functions from src 
include(srcdir("sparse_regress.jl"))
include(srcdir("vis_results.jl"))

# get data, originally from ==> https://www.nature.com/articles/s41467-019-12490-1 
files = readdir(datadir("exp_raw"))
data = matread(datadir("exp_raw", files[5]))

# load variables
du = data["dy"] ;
flag = data["flag"] # this is their way of identifying curve vs straightaway
v = data["v"]/10

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

# 3rd order library in V
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

    for i=1:n
        for j=i:n
            for k=j:n
                θ = hcat(θ, V[i,:].*V[j,:].*V[k,:])
            end
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
#display(p)

# model discovery for curve 
tol = 0.7
inds = flag_[goodinds].== 0. 
θcar = CarLib(V[:, goodinds][:, inds])
du_in = reshape(du[goodinds][inds], (length(du[goodinds][inds]),1))

Ξslic0, _ , ips_s0= EnAdSR(θcar, du_in, "slic", tol=tol, num_batches = 10)
Ξaicc0, _, ips_a0 = EnAdSR(θcar, du_in, "aicc", tol=tol, num_batches = 10)

Ξtrue0 = [2660., -57.5, 48.] # true programmed control law for curve
Ξtrue0 = vcat(Ξtrue0, zeros(length(Ξslic0)-3))
Ξtrue0 = reshape(Ξtrue0, (length(Ξtrue0), 1))

# visualize 
# SLIC curve
p1 = VisResults(Ξtrue0, Ξslic0, "SLIC Curve")
display(p1)
# AICc straightaway
p2 = VisResults(Ξtrue0, Ξaicc0, "AICc Curve")
display(p2)

# repeat for straightaway
tol = 0.7
inds = flag_[goodinds].== 1. 
θcar = CarLib(V[:, goodinds][:, inds])
du_in = reshape(du[goodinds][inds], (length(du[goodinds][inds]),1))

Ξslic1, _, ips_s1 = EnAdSR(θcar, du_in, "slic", tol=tol, num_batches = 10)
Ξaicc1, _, ips_a1 = EnAdSR(θcar, du_in, "aicc", tol=tol, num_batches = 10)

Ξtrue1 = [3610., -57.5, 48.] # true programmed control law for straightaway
Ξtrue1 = vcat(Ξtrue1, zeros(length(Ξslic1)-3))
Ξtrue1 = reshape(Ξtrue1, (length(Ξtrue1), 1))

# visualize
# SLIC straightaway
p1 = VisResults(Ξtrue1, Ξslic1, "SLIC Straightaway")
display(p1)
# AICc straightaway
p2 = VisResults(Ξtrue1, Ξaicc1, "AICc Straightaway")
display(p2)