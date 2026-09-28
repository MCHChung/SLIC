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
def save(fig,ax0,out,ncol=7,anchor=(1.004,0.5)):
    h,l=ax0.get_legend_handles_labels()
    fig.legend(h,l,loc='center left',bbox_to_anchor=anchor,fontsize=28,
               handlelength=0.9,handletextpad=0.3,labelspacing=0.25,borderpad=0.35)
    fig.savefig(out)
    for e in ('pdf','svg'): fig.savefig(out.rsplit('.',1)[0]+'.'+e)
    plt.close(fig); print("wrote",out.split('/')[-1])

# ================= WAIC / gMDL =================
SYS=[('lorenz','Lorenz'),('rossler','Rossler'),('lotka-volterra','Lotka-Volterra'),
     ('brusselator','Brusselator'),('van_der_pol','Van der Pol'),('nonlin_pendulum','Nonlinear pendulum')]
NO=[0,5,10,20,30,40]
COL={'slic':PCOLOR['slic'],'gmdl':'#1e9614','waic':'#1c1cc8','bic':'#888888'}
MK ={'slic':'o','gmdl':'s','waic':'v','bic':'P'}
LB ={'slic':'SLIC','gmdl':'gMDL','waic':'WAIC','bic':'BIC'}
def series(s,which,m):
    v=[]
    for ni in range(1,7):
        if which=='slic':
            p=glob.glob(f'{_R}/fig2_benchmarks/**/{s}_neffall_noise{ni}_results.jld',recursive=True)[0]
            key='Xis_neffall_slic'
        elif which=='bic':
            p=glob.glob(f'{_R}/fig2_benchmarks/**/{s}_neffall_noise{ni}_results.jld',recursive=True)[0]
            key='Xis_neffall_bic'
        else:
            p=glob.glob(f'{_R}/figS3_S4_waic_gmdl/**/{s}_waic_gmdl_noise{ni}.jld',recursive=True)[0]
            key=f'Xis_eff_{which}'
        with h5py.File(p,'r') as f: v.append(all_metrics(get(f,key),get(f,'Xitrue'))[m])
    return v
for metric,ylab,tag in [('f1','F1 score','f1'),('fpr','False-positive rate','fpr')]:
    fig,axes=plt.subplots(2,3,figsize=(18,10),sharey=True)
    for ax,(sysf,lbl) in zip(axes.ravel(),SYS):
        for w in ['slic','gmdl','waic','bic']:
            ax.plot(NO,series(sysf,w,metric),marker=MK[w],ms=13,color=COL[w],
                    lw=4.0 if w=='slic' else 2.6,label=LB[w],zorder=6 if w=='slic' else 2,
                    markeredgecolor='black',markeredgewidth=0.6)
        ax.set_title(lbl,fontsize=30,pad=10); ax.set_ylim(-0.04,1.06)
        ax.set_xticks([0,20,40]); ax.set_xlim(-3,43); ax.set_yticks([0,0.5,1.0])
        ax.tick_params(labelsize=24,pad=6)
    for ax in axes[1,:]: ax.set_xlabel('Noise level (%)',fontsize=28)
    for ax in axes[:,0]: ax.set_ylabel(ylab,fontsize=28)
    plt.tight_layout(); save(fig,axes[0,0],f'{_P}/SI_waic_gmdl_{tag}.png',ncol=4)
