import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import numpy as np, scipy.io as sio
from scipy.integrate import solve_ivp
R=str(EXP)+'/'
# ---------------- panel b: unforced run and forecast of the discovered model ----------------
d=sio.loadmat(R+'SloshingData.mat')['xData']; ts=np.ravel(d[0,0]); u=np.ravel(d[0,1]); dt=ts[1]-ts[0]
du=(u[:-4]/12 - 2*u[1:-3]/3 + 2*u[3:-1]/3 - u[4:]/12)/dt          # CalcDeriv, 4th-order central difference
tsteps=ts[2:-2]; x_exp=u[2:-2]
c1,c2,c3=-(7.802)**2, -2*0.072, 0.351                              # model displayed in Fig. 4b
sol=solve_ivp(lambda t,y:[y[1], c1*y[0]+c2*y[1]+c3*y[0]**3],(tsteps[0],tsteps[-1]),[x_exp[0],du[0]],t_eval=tsteps,rtol=1e-10,atol=1e-12)
x_pred=sol.y[0]; t_train=tsteps[299]                               # train_ind = 300 (1-based)
# ---------------- panels c, d: single-harmonic balance, as in freq_resp_curves.jl ----------------
Xi12,Xi22,Xi62=c1,c2,c3
alpha=Xi62/Xi12*1e4; eta=-(Xi22/2)/np.sqrt(-Xi12)
W=np.linspace(0.8,1.2,5000)
def slow_flow(uv,w,F):
    u_,v_=uv; R2=u_**2+v_**2
    # x = u cos(wt) + v sin(wt); balancing cos and sin terms with u'' and v'' neglected
    A=np.array([[2*eta, 2*w],[-2*w, 2*eta]])
    b=np.array([Fw:=F*w**2 - (2*eta*w*v_ + (1-w**2)*u_ + 0.75*alpha*R2*u_),
                -(-2*eta*w*u_ + (1-w**2)*v_ + 0.75*alpha*R2*v_)])
    return np.linalg.solve(A,b)
def branches(A_pct):
    F=0.3183*A_pct/100; pts=[]
    for w in W:
        # amplitude equation: [(1-w^2+3/4 a R^2)^2 + (2 eta w)^2] R^2 = (F w^2)^2, cubic in s = R^2
        a,b0=0.75*alpha,1-w**2
        coeffs=[a*a, 2*a*b0, b0*b0+(2*eta*w)**2, -(F*w*w)**2]
        for s in np.roots(coeffs):
            if abs(s.imag)>1e-12 or s.real<=0: continue
            s=s.real; Rr=np.sqrt(s); k=b0+a*s
            # u,v from the balance equations
            M=np.array([[k, 2*eta*w],[-2*eta*w, k]]); uv=np.linalg.solve(M,[F*w*w,0])
            J=np.zeros((2,2)); h=1e-7
            for j in range(2):
                e=np.zeros(2); e[j]=h; J[:,j]=(slow_flow(uv+e,w,F)-slow_flow(uv-e,w,F))/(2*h)
            stable=bool(np.all(np.linalg.eigvals(J).real<0))
            U,V=uv; phase=-2*np.degrees(np.arctan(V/(np.hypot(U,V)+U)))
            pts.append((w,100*Rr,phase,stable))
    return np.array(pts,dtype=float)
MEAS={A:np.loadtxt(R+f'DrivenSloshingTank/Measurements_A={A}%.txt',skiprows=1) for A in ['0.09','0.17','0.32']}
PRED={A:branches(float(A)) for A in MEAS}
