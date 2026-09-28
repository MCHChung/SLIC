"""Fig. 1b and SI Fig. S1 from one 6M-point Lorenz trajectory. n is increased by drawing
more rows from the same pool, never beyond 92% of it, so no sample is reused beyond what
sampling with replacement from 6M rows implies."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import numpy as np, time
from scipy.integrate import odeint
from scipy.signal import savgol_filter
def lorenz(u,t): x,y,z=u; return [10*(y-x), x*(28-z)-y, x*y-8/3*z]
LAB=['x','y','z']
def libcols(Y):
    n=Y.shape[0]; cols=[];nm=[]
    for i in range(n): cols.append(Y[i]); nm.append(LAB[i])
    for i in range(n):
        for j in range(i,n): cols.append(Y[i]*Y[j]); nm.append(LAB[i]+LAB[j])
    for i in range(n):
        for j in range(i,n):
            for k in range(j,n): cols.append(Y[i]*Y[j]*Y[k]); nm.append(LAB[i]+LAB[j]+LAB[k])
    return np.array(cols).T, nm
t0=time.time(); dt=1e-3; T=6000.
X=odeint(lorenz,[-8.,7.,27.],np.arange(0.,T+dt,dt),rtol=1e-10,atol=1e-11).T
rng=np.random.default_rng(17)
eta=5*np.mean(np.std(X,axis=1))/100
Xs=np.vstack([savgol_filter((X+eta*rng.standard_normal(X.shape))[i],61,3) for i in range(3)]); del X
dB=(-Xs[:,4:]+8*Xs[:,3:-1]-8*Xs[:,1:-3]+Xs[:,:-4])/(12*dt); Xc=Xs[:,2:-2]; del Xs
N=Xc.shape[1]; _,nm=libcols(Xc[:,:5]); P=len(nm)
sup=np.zeros((P,3),bool)
sup[[nm.index('x'),nm.index('y')],0]=True; sup[[nm.index('x'),nm.index('y'),nm.index('xz')],1]=True; sup[[nm.index('xy'),nm.index('z')],2]=True
xi=np.zeros((P,3)); xi[nm.index('x'),0]=-10; xi[nm.index('y'),0]=10
xi[nm.index('x'),1]=28; xi[nm.index('y'),1]=-1; xi[nm.index('xz'),1]=-1; xi[nm.index('xy'),2]=1; xi[nm.index('z'),2]=-8/3
k1=int(sup.sum()); k2=P*3
acc=0.
for s in range(0,N,400_000):
    Th,_=libcols(Xc[:,s:s+400_000]); acc+=np.sum((dB[:,s:s+400_000].T-Th@xi)**2)
sig=np.sqrt(acc/(N*3)); print(f"pool N={N}, matched sd {sig:.4f}, {time.time()-t0:.0f}s",flush=True)
def stats(idx,arm,rg):
    G=np.zeros((P,P)); b=np.zeros((P,3)); yy=np.zeros(3)
    for s in range(0,len(idx),400_000):
        ii=idx[s:s+400_000]; Th,_=libcols(Xc[:,ii])
        y=(Th@xi + sig*rg.standard_normal((len(ii),3))) if arm=='A' else dB[:,ii].T
        G+=Th.T@Th; b+=Th.T@y; yy+=np.einsum('ij,ij->j',y,y)
    return G,b,yy
def sse(G,b,yy,cols=None):
    if cols is None: Xo=np.linalg.solve(G,b); return float(yy.sum()-2*np.sum(Xo*b)+np.sum(Xo*(G@Xo)))
    t=0.
    for e in range(3):
        c=cols[:,e]; Xe=np.linalg.solve(G[np.ix_(c,c)],b[c,e]); t+=float(yy[e]-2*Xe@b[c,e]+Xe@(G[np.ix_(c,c)]@Xe))
    return t
NS=[100,500,1000,5000,10000,50000,100000,500000,1000000,2000000,5000000]
assert max(NS)<=0.92*N
res={'A':[],'B':[]}
for n in NS:
    rg=np.random.default_rng(1000+n); idx=rg.integers(0,N,n)
    for arm in 'AB':
        G,b,yy=stats(idx,arm,rg); lr=np.log((sse(G,b,yy)/n)/(sse(G,b,yy,sup)/n))
        res[arm].append((n,lr,lr+2*(k2-k1)/n,lr+(k2-k1)*np.log(n)/n,lr+np.log(k2/k1)))
    print(f"n={n:>8}  B: AIC {res['B'][-1][2]:+.5f} BIC {res['B'][-1][3]:+.5f} SLIC {res['B'][-1][4]:+.4f} | A: AIC {res['A'][-1][2]:+.5f} BIC {res['A'][-1][3]:+.5f}  ({time.time()-t0:.0f}s)",flush=True)
np.save(RESULTS/'fig1b_figS1'/'fig1b_final.npy',{'A':res['A'],'B':res['B'],'k1':k1,'k2':k2,'N':N},allow_pickle=True)
print("DONE",flush=True)
