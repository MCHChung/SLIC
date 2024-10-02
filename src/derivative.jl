# 4th order scheme for differentiating once along columns 
function CalcDeriv(u,dt)
    #  u: Measurement data we wish to approxiamte the derivative. 
    #  It should be of size n x m, where n is the number of measurement, m is the number of states.
    # dt: Time step
    # return du: The approximated derivative. 
    #--------------------------- 
    
    # Define the coeficient for different orders of derivative
    p1=1/12;p2=-2/3;p3=0;p4=2/3;p5=-1/12;
    
    du=(p1*u[1:end-4,:]+p2*u[2:end-3,:]+p3*u[3:end-2,:]+p4*u[4:end-1,:]+p5*u[5:end,:])/dt;
        
    return du
end

function DataWithFirstDeriv(u::AbstractMatrix, ts::AbstractVector, dt)
    du = CalcDeriv(u',dt)'
    return ts[3:end-2], u[:, 3:end-2], du # return: trimmed tsteps, trimmmed state vec data, and deriv estimate 
end