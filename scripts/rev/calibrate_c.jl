using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics, LinearAlgebra, Combinatorics, Printf

include(scriptsdir("rev", "rev_common.jl"))

# ----------------------------------------------------------------------------
# Note on the residual floor eta (applies identically to every criterion,
# SLIC included). Candidates are scored on RSS + eta rather than RSS; see
# score() in src/sparse_regress.jl. In the main pipeline eta = c*cond(theta),
# with the per-system constant c passed to EnAdSR. In the exhaustive
# enumeration (enumerate_all.jl) eta = gamma*RSS0, where RSS0 is the true
# model's residual on noise-free data (calibrate_c.jl) and gamma = 100
# (SLIC_GAMMA). The floor is numerical: as the noise goes to zero, RSS goes to
# zero, which would otherwise make the log-likelihood terms diverge and leave
# the fit term free to fall with every added term, so that most criteria would
# not sparsify at low noise.
# ----------------------------------------------------------------------------


# ============================================================================
# Calibrate the conditioning weight c on NOISE-FREE data.
#
# Rationale. At NoisePct=0 there is no measurement noise, but the weak-form
# projection leaves a deterministic approximation error. That residual is a
# smooth function of the state, hence partially absorbable by spurious library
# columns. Criteria accept a term iff it cuts RSS by more than a threshold:
#
#     BIC  ~ exp(log n / n) - 1   ~ 1.4%   at n_eff ~ 442
#     AIC  ~ exp(2 / n) - 1       ~ 0.45%
#     SLIC ~ (k+1)/k - 1          ~ 33-50%
#
# so fit-weighted criteria buy the spurious columns and SLIC does not. The
# conditioning floor η = c·cond(θ) suppresses this, but only once η is
# comparable to (or larger than) the systematic residual it must mask.
#
# This script sweeps c at noise=0 and reports, per criterion and equation:
#   - whether the exact true support is recovered
#   - the number of terms selected (to see over- AND under-selection)
#   - η / RSS(true support), the floor-to-systematic-error ratio
#
# The calibrated c is the SMALLEST value at which every criterion recovers the
# exact model on noise-free data. That is a calibration against a known ground
# truth in a condition where all criteria should agree; it is deliberately
# generous to the competitors, so any degradation at higher noise cannot be
# attributed to an unfavourable floor.
#
# Noise-free inputs are deterministic, so ONE run suffices (no noise
# realisation to average over). The whole sweep runs in seconds to minutes.
#
# Usage (inside a salloc devel session):
#     SLIC_CAL_SYS=1 julia --project=. scripts/rev/calibrate_c.jl
#   or loop:  for s in 1 2 3 4 5 6; do SLIC_CAL_SYS=$s julia --project=. scripts/rev/calibrate_c.jl; done
# ============================================================================

const ICS_CAL = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]

# log-spaced grid spanning the plausible range (main-text c values are 1e-3..1e0)
const C_GRID = haskey(ENV, "SLIC_CAL_CGRID") ?
    parse.(Float64, split(ENV["SLIC_CAL_CGRID"], ",")) :
    [10.0^e for e in -4.0:0.5:1.5]   # 12 points, spans the main-text range

# --- degree-2 helpers (Lorenz/Rossler enumerate over the degree-2 library) ---
# LorLib has NO bias  -> degree-2 block = columns 1:9
# RossLib HAS a bias  -> degree-2 block = columns 1:10
degree2_ncols(sys::Int) = sys == 1 ? 9 : 10

function degree2_xitrue(sys::Int)
    if sys == 1  # Lorenz, 9 cols: [x y z x² xy xz y² yz z²]
        Ξ = zeros(9, 3)
        Ξ[1, 1] = -10.0; Ξ[2, 1] = 10.0
        Ξ[1, 2] = 28.0;  Ξ[2, 2] = -1.0; Ξ[6, 2] = -1.0
        Ξ[5, 3] = 1.0;   Ξ[3, 3] = -8/3
        return Ξ
    elseif sys == 2  # Rossler, 10 cols: [1 x y z x² xy xz y² yz z²]
        Ξ = zeros(10, 3)
        Ξ[3, 1] = -1.0;  Ξ[4, 1] = -1.0
        Ξ[2, 2] = 1.0;   Ξ[3, 2] = 0.2
        Ξ[1, 3] = 0.2;   Ξ[7, 3] = 1.0;  Ξ[4, 3] = -5.7
        return Ξ
    else
        error("degree2_xitrue only for sys 1 or 2")
    end
end

"""Relative RSS reduction from adding the single best spurious column."""
function best_spurious_reduction(θ, y, true_supp)
    Xs = θ[:, true_supp]
    b = Xs \ y
    rss0 = sum(abs2, y - Xs * b)
    best = 0.0
    bestj = 0
    for j in axes(θ, 2)
        j in true_supp && continue
        Xa = θ[:, vcat(true_supp, j)]
        ba = Xa \ y
        rss1 = sum(abs2, y - Xa * ba)
        red = (rss0 - rss1) / max(rss0, eps())
        if red > best
            best = red; bestj = j
        end
    end
    return rss0, best, bestj
end

function main(sys::Int)
    fname, sysname, n_state, n_lib_full, dt, Tdef = SYSTEMS[sys]
    data = load(datadir("sims", "ode_data", fname))

    # Noise-free inputs (deterministic; one evaluation suffices)
    qt, θ_full, n_eff = GetInputsWithNeff(sys, data["Xtrues"], data["ts"], LIBS[sys], 0)

    if sys in (1, 2)
        θ = θ_full[:, 1:degree2_ncols(sys)]
        Ξtrue = degree2_xitrue(sys)
        libnote = "degree-2 subset ($(degree2_ncols(sys)) cols)"
    else
        θ = θ_full
        Ξtrue = data["Ξtrue"]
        libnote = "full library ($(size(θ,2)) cols)"
    end

    n_lib = size(θ, 2)
    n_obs = size(θ, 1)
    cn = cond(θ)

    println("="^78)
    println("c-CALIBRATION at noise=0 :  $sysname (sys=$sys), $libnote")
    println("  n_obs=$n_obs  n_eff=$(round(n_eff, digits=1))  cond(θ)=$(round(cn, sigdigits=4))")
    # Per-criterion acceptance thresholds, for context
    @printf("  per-term RSS-cut threshold:  BIC %.3f%%   AIC %.3f%%   SLIC(k=2) %.1f%%\n",
            (exp(log(n_eff)/n_eff) - 1)*100, (exp(2/n_eff) - 1)*100, 50.0)

    # Diagnostic: how absorbable is the noise-free residual?
    println("\n  Residual structure on the true support (noise-free):")
    for eq = 1:n_state
        y = qt'[:, eq]
        ts_supp = findall(abs.(Ξtrue[:, eq]) .> 0)
        rss0, red, j = best_spurious_reduction(θ, y, ts_supp)
        @printf("    eq%d: RSS=%.4g   best spurious col %d cuts RSS by %.2f%%\n", eq, rss0, j, red*100)
    end
    println("    (a cut above the BIC threshold but below SLIC's explains the zero-noise gap)")
    flush(stdout)

    # RSS on true support, per equation, used for the η/RSS ratio
    rss_true = [begin
        y = qt'[:, eq]; s = findall(abs.(Ξtrue[:, eq]) .> 0)
        Xs = θ[:, s]; sum(abs2, y - Xs * (Xs \ y))
    end for eq = 1:n_state]

    println("\n  Sweep over c   (E = exact recovery, number = terms selected; true counts: " *
            join([string(count(abs.(Ξtrue[:, e]) .> 0)) for e = 1:n_state], "/") * ")")
    hdr = @sprintf("  %-10s %-9s", "c", "η/RSS")
    for ic in ICS_CAL
        hdr *= @sprintf("%-9s", uppercase(ic))
    end
    println(hdr)

    results = Dict{Float64, Any}()
    for c in C_GRID
        tstart = time()
        η = c * cn

        # ONE enumeration pass per equation returns every criterion's pick.
        # (The OLS fits dominate the cost and are identical across criteria.)
        nsel_by = Dict(ic => Int[] for ic in ICS_CAL)
        ex_by   = Dict(ic => Bool[] for ic in ICS_CAL)
        for eq = 1:n_state
            y = reshape(qt'[:, eq], (size(qt, 2), 1))
            Ξd, _, _ = enumerate_supports(θ, y, ICS_CAL;
                                          n_eff=n_eff, max_k=nothing,
                                          η=η, verbose=false)
            tru = findall(abs.(Ξtrue[:, eq]) .> 0)
            for ic in ICS_CAL
                sel = findall(abs.(Ξd[ic][:, 1]) .> 0)
                push!(nsel_by[ic], length(sel))
                push!(ex_by[ic], sel == tru)
            end
        end

        exact_all = all(all(ex_by[ic]) for ic in ICS_CAL)
        n_true = [count(abs.(Ξtrue[:, e]) .> 0) for e = 1:n_state]
        under_any = any(any(nsel_by[ic] .< n_true) for ic in ICS_CAL)

        cells = String[]
        percrit = Dict{String, Any}()
        for ic in ICS_CAL
            percrit[ic] = (all(ex_by[ic]), nsel_by[ic])
            push!(cells, (all(ex_by[ic]) ? "E" : "·") * join(string.(nsel_by[ic]), ""))
        end

        ratio = η / mean(rss_true)
        line = @sprintf("  %-10.3g %-9.2f", c, ratio)
        for s_ in cells
            line *= @sprintf("%-9s", s_)
        end
        line *= exact_all ? "  <== ALL EXACT" : ""
        line *= under_any ? "  (under-selection)" : ""
        line *= @sprintf("   [%.1fs]", time() - tstart)
        println(line)
        results[c] = (exact_all=exact_all, under_any=under_any, percrit=percrit, ratio=ratio)
        flush(stdout)
    end

    # Smallest c with all criteria exact and no under-selection
    ok = sort([c for c in C_GRID if results[c].exact_all && !results[c].under_any])
    if isempty(ok)
        println("\n  NO c in the grid recovers the exact model for every criterion.")
        println("  Widen C_GRID, or accept per-criterion calibration.")
    else
        cmin = first(ok); cmax = last(ok)
        @printf("\n  CALIBRATED c = %.3g   (window of all-exact c: %.3g … %.3g, η/RSS %.1f … %.1f)\n",
                cmin, cmin, cmax, results[cmin].ratio, results[cmax].ratio)
        @printf("  main-text c for this system = %s\n",
                sys == 1 ? "1e-1" : sys == 2 ? "1e-1" : sys == 3 ? "1e-3" :
                sys == 4 ? "1e-2" : sys == 5 ? "1e0" : "1e-2")
    end

    outdir = datadir("sims", "ode_results_rev", "c_calibration")
    mkpath(outdir)
    wsave(joinpath(outdir, "$(lowercase(replace(sysname, " " => "_")))_ccal.jld"),
          Dict{String,Any}("sys"=>sys, "sysname"=>sysname, "c_grid"=>collect(C_GRID),
                           "cond_theta"=>cn, "n_eff"=>n_eff, "rss_true"=>rss_true,
                           "exact_all"=>[results[c].exact_all for c in C_GRID],
                           "under_any"=>[results[c].under_any for c in C_GRID]))
    println("\n  wrote $(outdir)")
end

sys = parse(Int, ENV["SLIC_CAL_SYS"])
main(sys)
