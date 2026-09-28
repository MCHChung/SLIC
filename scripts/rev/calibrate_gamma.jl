using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics, LinearAlgebra, Combinatorics, Printf

include(scriptsdir("rev", "rev_common.jl"))

# ============================================================================
# Calibrate a DIMENSIONLESS conditioning floor.
#
# Motivation. The absolute floor η = c·cond(θ) is shared across the equations
# of a system, but the noise-free residual it must mask differs per equation by
# up to 1e3 (Rossler) or 1e11 (Van der Pol, whose dx/dt = v is reproduced
# exactly by the weak form). A single c therefore cannot sit inside every
# equation's admissible window; Rossler's window collapsed to one grid point.
#
# Replace it with a per-equation floor scaled by that equation's own systematic
# residual:
#
#     η_eq = γ · RSS⁽⁰⁾_eq        (RSS⁽⁰⁾ = noise-free residual, true support)
#
# γ is dimensionless and, if the mechanism is right, should be roughly
# universal across systems. cond(θ) drops out entirely.
#
# Reading: a criterion must not try to explain residual structure lying within
# a factor γ of the derivative estimator's own approximation error.
#
# Usage (batch, one task per system):
#     sbatch --array=0-5 cluster/submit_calibrate_gamma.sh
# ============================================================================

const ICS_CAL = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]

const GAMMA_GRID = haskey(ENV, "SLIC_GAMMA_GRID") ?
    parse.(Float64, split(ENV["SLIC_GAMMA_GRID"], ",")) :
    [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4]

degree2_ncols(sys::Int) = sys == 1 ? 9 : 10

function degree2_xitrue(sys::Int)
    if sys == 1
        Ξ = zeros(9, 3)
        Ξ[1, 1] = -10.0; Ξ[2, 1] = 10.0
        Ξ[1, 2] = 28.0;  Ξ[2, 2] = -1.0; Ξ[6, 2] = -1.0
        Ξ[5, 3] = 1.0;   Ξ[3, 3] = -8/3
        return Ξ
    elseif sys == 2
        Ξ = zeros(10, 3)
        Ξ[3, 1] = -1.0;  Ξ[4, 1] = -1.0
        Ξ[2, 2] = 1.0;   Ξ[3, 2] = 0.2
        Ξ[1, 3] = 0.2;   Ξ[7, 3] = 1.0;  Ξ[4, 3] = -5.7
        return Ξ
    else
        error("degree2_xitrue only for sys 1 or 2")
    end
end

function main(sys::Int)
    fname, sysname, n_state, n_lib_full, dt, Tdef = SYSTEMS[sys]
    data = load(datadir("sims", "ode_data", fname))

    qt, θ_full, n_eff = GetInputsWithNeff(sys, data["Xtrues"], data["ts"], LIBS[sys], 0)

    if sys in (1, 2)
        θ = θ_full[:, 1:degree2_ncols(sys)]
        Ξtrue = degree2_xitrue(sys)
    else
        θ = θ_full
        Ξtrue = data["Ξtrue"]
    end

    # Noise-free residual of the TRUE model, per equation. This is the
    # systematic (projection / quadrature) error the floor must exceed.
    rss0 = [begin
        y = qt'[:, eq]
        s = findall(abs.(Ξtrue[:, eq]) .> 0)
        Xs = θ[:, s]
        sum(abs2, y - Xs * (Xs \ y))
    end for eq = 1:n_state]

    println("="^80)
    println("γ-CALIBRATION at noise=0 :  $sysname (sys=$sys)")
    println("  n_eff=$(round(n_eff, digits=1))   n_lib=$(size(θ,2))")
    @printf("  per-term RSS-cut threshold:  BIC %.3f%%   AIC %.3f%%   SLIC(k=2) 50%%\n",
            (exp(log(n_eff)/n_eff) - 1)*100, (exp(2/n_eff) - 1)*100)
    println("  noise-free RSS per equation: ", [round(r, sigdigits=4) for r in rss0])
    println("  true term counts:            ", [count(abs.(Ξtrue[:, e]) .> 0) for e = 1:n_state])
    flush(stdout)

    println("\n  E = exact for that equation, digit = #terms selected")
    hdr = @sprintf("  %-9s", "gamma")
    for ic in ICS_CAL
        hdr *= @sprintf("%-10s", uppercase(ic))
    end
    println(hdr)

    per_eq_ok = Dict(eq => Float64[] for eq = 1:n_state)
    for γ in GAMMA_GRID
        t0 = time()
        cells = String[]
        exact_by_eq = Dict(eq => true for eq = 1:n_state)
        nsel_store = Dict(ic => Int[] for ic in ICS_CAL)
        ex_store   = Dict(ic => Bool[] for ic in ICS_CAL)

        for eq = 1:n_state
            η_eq = γ * rss0[eq]
            y = reshape(qt'[:, eq], (size(qt, 2), 1))
            Ξd, _, _ = enumerate_supports(θ, y, ICS_CAL;
                                          n_eff=n_eff, max_k=nothing,
                                          η=η_eq, verbose=false)
            tru = findall(abs.(Ξtrue[:, eq]) .> 0)
            for ic in ICS_CAL
                sel = findall(abs.(Ξd[ic][:, 1]) .> 0)
                push!(nsel_store[ic], length(sel))
                ex = (sel == tru)
                push!(ex_store[ic], ex)
                ex || (exact_by_eq[eq] = false)
            end
        end

        for eq = 1:n_state
            exact_by_eq[eq] && push!(per_eq_ok[eq], γ)
        end

        for ic in ICS_CAL
            push!(cells, (all(ex_store[ic]) ? "E" : "·") * join(string.(nsel_store[ic]), ""))
        end
        line = @sprintf("  %-9.4g", γ)
        for s_ in cells
            line *= @sprintf("%-10s", s_)
        end
        allex = all(all(ex_store[ic]) for ic in ICS_CAL)
        line *= allex ? "  <== ALL EXACT" : ""
        line *= @sprintf("   [%.1fs]", time() - t0)
        println(line)
        flush(stdout)
    end

    # per-equation admissible windows, then the intersection
    println("\n  per-equation admissible γ (all criteria exact):")
    for eq = 1:n_state
        w = per_eq_ok[eq]
        println("    eq$eq: ", isempty(w) ? "none in grid" : "$(minimum(w)) … $(maximum(w))")
    end
    inter = intersect([Set(per_eq_ok[eq]) for eq = 1:n_state]...)
    if isempty(inter)
        println("\n  NO γ works for every equation in this grid. Widen SLIC_GAMMA_GRID.")
    else
        g = sort(collect(inter))
        @printf("\n  SYSTEM-WIDE γ window: %.4g … %.4g   (recommend the geometric centre: %.4g)\n",
                first(g), last(g), sqrt(first(g) * last(g)))
    end

    outdir = datadir("sims", "ode_results_rev", "gamma_calibration")
    mkpath(outdir)
    wsave(joinpath(outdir, "$(lowercase(replace(sysname, " " => "_")))_gcal.jld"),
          Dict{String,Any}("sys"=>sys, "sysname"=>sysname,
                           "gamma_grid"=>collect(GAMMA_GRID),
                           "rss0"=>rss0, "n_eff"=>n_eff,
                           "per_eq_ok"=>Dict(string(k)=>v for (k,v) in per_eq_ok)))
    println("\n  wrote $(outdir)")
end

main(parse(Int, ENV["SLIC_CAL_SYS"]))
