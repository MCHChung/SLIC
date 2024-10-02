using HarmonicBalance, Plots, DelimitedFiles, LaTeXStrings

# set defaults 
default(dpi=300, grid=false, fontfamily="computer modern")

# load data 
files = readdir(datadir("exp_raw", "DrivenSloshingTank"))
dd1 = readdlm(datadir("exp_raw", "DrivenSloshingTank", files[1])) # A = 0.09%, driving amp as % of tank width
dd2 = readdlm(datadir("exp_raw", "DrivenSloshingTank", files[2])) # A = 0.17%
dd3 = readdlm(datadir("exp_raw", "DrivenSloshingTank", files[3])) # A = 0.32%

# The following function plots the frequency-response curves *assuming a rescaled Duffing model!*
function PlotFreqRespCurves(Ξ)
    @variables α, ω, ω0, F, t, η, x(t) # declare constant variables and a function x(t)
    # this just builds the Duffing Equation
    diff_eq = DifferentialEquation(d(x,t,2) + ω0^2*x + α*x^3 + 2*η*d(x,t) ~ F*ω^2*cos(ω*t), x)

    add_harmonic!(diff_eq, x, ω) # specify the ansatz x = u(T) cos(ωt) + v(T) sin(ωt)

    # implement ansatz to get harmonic equations
    harmonic_eq = get_harmonic_equations(diff_eq)

    # Here we assume we rescale all of the variables: 
        # ω0 = sqrt(-Ξ[1,2]) -> 1 
        # η -> -(Ξ[2,2]/2)/ω0
        # α = ->  -(Ξ[6,2]/ω0^2)*10^4, where the 10^4 comes from data rescaling of original data 
        # F is the varying driving amplitude from the experimental data 
    fixed1 = (α => Ξ[6,2]/Ξ[1,2]*10^4, ω0 => 1.0, F => 0.3183*0.09/100, η=> -(Ξ[2,2]/2)/sqrt(-Ξ[1,2]))   # fixed parameters
    fixed2 = (α => Ξ[6,2]/Ξ[1,2]*10^4, ω0 => 1.0, F => 0.3183*0.17/100, η=> -(Ξ[2,2]/2)/sqrt(-Ξ[1,2]))   # fixed parameters
    fixed3 = (α => Ξ[6,2]/Ξ[1,2]*10^4, ω0 => 1.0, F => 0.3183*0.32/100, η=> -(Ξ[2,2]/2)/sqrt(-Ξ[1,2]))   # fixed parameters

    varied = ω => LinRange(0.8, 1.2, 5000)           # range of parameter values
    result1 = get_steady_states(harmonic_eq, varied, fixed1)
    result2 = get_steady_states(harmonic_eq, varied, fixed2)
    result3 = get_steady_states(harmonic_eq, varied, fixed3)

    # make amplitude plot 
    p_amp = plot(result1, "100*sqrt(abs(u1)^2 + abs(v1)^2)",  color=1, legend=:topright, label="Pred: A=0.09%")
    scatter!(dd1[2:end , 1],dd1[2:end , 2], color=1, label="Pred: A=0.09%")
    
    plot!(result2, "100*sqrt(abs(u1)^2 + abs(v1)^2)",  color=2, label="Pred: A=0.17%")
    scatter!(dd2[2:end , 1],dd2[2:end , 2], color=2, label="Pred: A=0.17%")
    
    plot!(result3, "100*sqrt(abs(u1)^2 + abs(v1)^2)", color=3, label="Pred: A=0.32%")
    scatter!(dd3[2:end , 1],dd3[2:end , 2], color=3, label="Pred: A=0.32%")
    plot!(legend=false)
    
    ylims!(0,6)
    ylabel!("Amplitude (%)")
    xlabel!(LaTeXString("Ω/\$ ω_1 \$"))

    # make phase plot 
    p_phase = plot(result1, "-2*atand(v1/(sqrt(abs(u1)^2 + abs(v1)^2) + u1))", legend=false, color=1)
    scatter!(dd1[2:end , 1],dd1[2:end , 3], color=1)

    plot!(result2, "-2*atand(v1/(sqrt(abs(u1)^2 + abs(v1)^2) + u1))", legend=false, color=2)
    scatter!(dd2[2:end , 1],dd2[2:end , 3], color=2)

    plot!(result3, "-2*atand(v1/(sqrt(abs(u1)^2 + abs(v1)^2) + u1))", legend=false, color=3)
    scatter!(dd3[2:end , 1],dd3[2:end , 3], color=3)

    ylims!(-180, 0)
    ylabel!("Phase shift (deg)")
    xlabel!(LaTeXString("Ω/\$ ω_1 \$"))
    
    return p_amp, p_phase
end
