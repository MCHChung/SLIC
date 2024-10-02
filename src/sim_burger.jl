using DifferentialEquations , FFTW, Plots, Measures

# defines time derivative of burger's eqn that we solve via FFT 
function dudt_bg(du, u, ps, t)
    d = ps[1:9] # model coefficients 
    k = ps[10:end] # fourier freqs

    con_hat = fft(ones(size(u)))

    uhat = fft(u)
    uhatsq = fft(u.^2)
    uhatcb = fft(u.^3)

    uxhat = @. im*k*uhat
    uxxhat = @. -k^2 * uhat
    uxxxhat = @. -im*k^3 * uhat 
    uxxxxhat = @. k^4 * uhat
    uuxhat = @. 0.5*im*k*uhatsq
    
    # compute deriv using model 
    # model terms on RHS: 1, u, u_x , u_xx , u_xxx, u_xxxx, u^2, u*u_x, u^3
    du_hat = d[1]*con_hat + d[2]*uhat + d[3]*uxhat + d[4]*uxxhat + d[5]*uxxxhat + d[6]*uxxxxhat + d[7]*uhatsq + d[8]*uuxhat + d[9]*uhatcb
    du .= real(ifft(du_hat))
end

# solves burgers equation 
# u0: initial cond 
# tspan: time span of sim 
# dt: time step 
# ps: coeffcient matrices 
function SimBurg(u0, tspan, dt, ps)
    ptrue, paic, pslic = ps[1], ps[2], ps[3]
    prob = ODEProblem(dudt_bg , u0, tspan, ptrue)
    prob_aic = remake(prob, p=paic)
    prob_slic = remake(prob, p=pslic)

    soln = solve(prob, Tsit5(), saveat=dt)
    soln_aic = solve(prob_aic, Tsit5(), saveat=dt)
    soln_slic = solve(prob_slic, Tsit5(), saveat=dt)

    usol_true = Array(soln) ;
    usol_aic = Array(soln_aic) ;
    usol_slic = Array(soln_slic) ; 

    return usol_true, usol_aic, usol_slic
end

# plots the solutions as heatmaps
function PlotBurg(ts, xs, utrue, uaic, uslic)
    p1 = heatmap(ts ,xs, utrue)
    title!("True")
    xlabel!("t")
    ylabel!("x")

    p2 = heatmap(ts,xs,uslic)
    title!("SLIC")
    xlabel!("t")
    ylabel!("x")

    p3 = heatmap(ts,xs,uaic)
    title!("AICc")
    xlabel!("t")
    ylabel!("x")

    return plot(p1,p2,p3, layout=(1,3), size=(1500,300), titlefontsize=12, margins=12*Measures.mm)
end