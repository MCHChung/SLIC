using DrWatson, MAT, JLD, Plots
@quickactivate "SLIC"

files = readdir(datadir("sims", "pde_data"))
#datadir("sims", "pde_data", files[1])
u = matread(datadir("sims", "pde_data", files[4]))

keys(u)

files = readdir(datadir("sims", "ode_results_main"))
x = load(datadir("sims", "ode_results_main", files[2]))
#plot(x["Xtrue"]', x["DXtrue"]')

x["Ξtrue"]
