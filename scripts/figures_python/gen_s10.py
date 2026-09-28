import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, PLOTS
_R, _P = str(RESULTS), str(PLOTS)
import h5py, numpy as np, glob, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paperstyle import apply_paper_style, PCOLOR, PLW, PMS, PMARK, PORDER, PLABEL
from mlib import get, all_metrics
apply_paper_style()
def dd(f,dref):
    ks=[x.decode() if isinstance(x,bytes) else x for x in f[dref['keys_']][:]]
    return dict(zip(ks,f[dref['values_']]))
def deref(f,ref):
    o=f[ref]; v=o[()]
    if hasattr(v,'dtype') and v.dtype.names and 'parent_' in (v.dtype.names or ()): return f[v['parent_']][:]
    return v
SYS=[('lorenz','Lorenz'),('rossler','Rossler'),('lotka-volterra','Lotka-Volterra'),
     ('brusselator','Brusselator'),('van_der_pol','Van der Pol'),('nonlin_pendulum','Nonlinear pendulum')]
NO=[0,5,10,20,30,40]
def frame(nr,nc,figw,figh):
    fig,axes=plt.subplots(nr,nc,figsize=(figw,figh),squeeze=False)
    return fig,axes
def finish(fig,ax0,out,ylab):
    h,l=ax0.get_legend_handles_labels()
    fig.legend(h,l,loc='center left',bbox_to_anchor=(1.004,0.5),fontsize=30,
               handlelength=0.9,handletextpad=0.3,labelspacing=0.25,borderpad=0.35)
    fig.savefig(out)
    for e in ('pdf','svg'): fig.savefig(out.rsplit('.',1)[0]+'.'+e)
    plt.close(fig); print("wrote",out.split('/')[-1])

# ---------- LONG TRAJECTORY ----------
def m_at(path,noise,ic,m):
    with h5py.File(path,'r') as f:
        Xt=get(f,'Xitrue'); res=dd(f,f['results'][()])
        if noise not in res: return np.nan
        inner=dd(f,f[res[noise]][()]); xis=dd(f,f[inner['Xis']][()])
        return all_metrics(deref(f,xis[ic]),Xt)[m]
fig,axes=frame(2,3,18,10)
for ax,(sysf,lbl) in zip(axes.ravel(),SYS):
    files=sorted(glob.glob(f'{_R}/figS10_long_trajectory/**/{sysf}_L*_results.jld',recursive=True),
                 key=lambda p:int(re.search(r'_L(\d+)_',p).group(1)))
    Ls=[int(re.search(r'_L(\d+)_',p).group(1)) for p in files]
    for ic in PORDER:
        ax.plot(Ls,[m_at(p,20,ic,'acc') for p in files],marker=PMARK[ic],ms=PMS[ic]-3,
                color=PCOLOR[ic],lw=PLW[ic]-0.8,label=PLABEL[ic],
                zorder=6 if ic=='slic' else 2,markeredgecolor='black',markeredgewidth=0.6)
    ax.set_xscale('log'); ax.set_xticks(Ls); ax.set_xticklabels(Ls); ax.minorticks_off()
    ax.set_ylim(-0.04,1.06); ax.set_yticks([0,0.5,1.0])
    ax.set_title(lbl,fontsize=30,pad=10); ax.tick_params(labelsize=24,pad=6)
for ax in axes[1,:]: ax.set_xlabel('Trajectory length factor $L$',fontsize=28)
for ax in axes[:,0]: ax.set_ylabel('Accuracy (20% noise)',fontsize=28)
plt.tight_layout(); finish(fig,axes[0,0],f'{_P}/SI_long_trajectory.png','acc')
