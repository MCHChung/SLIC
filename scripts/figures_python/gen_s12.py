import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, PLOTS
_R, _P = str(RESULTS), str(PLOTS)
"""SI enumeration figure (Supp. Fig. S12): exhaustive support enumeration on a
FIXED candidate set, each criterion ranking the same supports. Removes the
candidate-generation loop confound entirely."""
import h5py, glob, numpy as np, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paperstyle import apply_paper_style, PCOLOR, PLW, PMS, PMARK, PORDER, PLABEL
apply_paper_style()
ROOT=f'{_R}/figS12_enumeration'
NO=[0,5,10,20,30,40]
SYSN={'lorenz':'Lorenz','rossler':'Rossler','lotka-volterra':'Lotka-Volterra',
      'brusselator':'Brusselator','van_der_pol':'Van der Pol','nonlin_pendulum':'Nonlinear pendulum'}

def read(fn):
    with h5py.File(fn,'r') as h:
        o=h['Xis_eff_eq'][()]
        keys=[k.decode() if isinstance(k,bytes) else k for k in h[o[0]][()]]
        vals=[np.asarray(h[v][()]) if not isinstance(v,np.ndarray) else v for v in h[o[1]][()]]
        xt=np.asarray(h['Xitrue_col'][()])
        return dict(zip(keys,vals)), xt, int(h['n_supports'][()])

# per system: exact-recovery of the equation, averaged over equations and runs
avail=sorted({re.match(r'(.+)_enum_noise\d+_eq\d+\.jld',f.split('/')[-1]).group(1)
              for f in glob.glob(f'{ROOT}/**/*.jld',recursive=True)})
avail=[s for s in avail if s in SYSN]
print("systems:",avail)
res={}
for s in avail:
    for ic in PORDER:
        row=[]
        for ni in range(1,7):
            fs=sorted(glob.glob(f'{ROOT}/**/{s}_enum_noise{ni}_eq*.jld',recursive=True))
            if not fs: row.append(np.nan); continue
            # whole-system recovery: every equation correct in the SAME run,
            # matching the system-level recovery metric used in Fig. 2
            per_eq=[]
            for fn in fs:
                D,xt,ns_=read(fn)
                if ic not in D: continue
                A=D[ic]; tnz=np.abs(xt)>1e-12
                per_eq.append([bool(np.all((np.abs(A[r])>1e-12)==tnz)) for r in range(A.shape[0])])
            if not per_eq: row.append(np.nan); continue
            nr=min(len(e) for e in per_eq)
            row.append(float(np.mean([all(e[r] for e in per_eq) for r in range(nr)])))
        res[(s,ic)]=row

n=len(avail); ncol=3; nrow=int(np.ceil(n/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(18,5.2*nrow),sharex=True,sharey=True,squeeze=False)
for ax,s in zip(axes.ravel(),avail):
    for ic in PORDER:
        ax.plot(NO,res[(s,ic)],marker=PMARK[ic],ms=PMS[ic],color=PCOLOR[ic],lw=PLW[ic],
                label=PLABEL[ic],zorder=6 if ic=='slic' else 2,
                markeredgecolor='black',markeredgewidth=0.8)
    ax.set_title(SYSN[s],pad=10); ax.set_ylim(-0.04,1.06)
    ax.set_yticks([0,0.5,1.0]); ax.set_xticks([0,20,40]); ax.set_xlim(-3,43)
    ax.tick_params(pad=8,labelsize=24)
for ax in axes.ravel()[n:]: ax.axis('off')
for ax in axes[-1,:]: ax.set_xlabel('Noise level (%)')
for ax in axes[:,0]: ax.set_ylabel('Correct support\nselected',fontsize=27,labelpad=12)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='center left',bbox_to_anchor=(1.005,0.5),handlelength=1.0,
           handletextpad=0.35,labelspacing=0.28,borderpad=0.35)
plt.tight_layout()
plt.savefig(f'{_P}/SI_enumeration.png')
[plt.savefig(f'{_P}/SI_enumeration.'+_e) for _e in ('pdf','svg')]  # VECTOR
print("\nCorrect-support selection from the FIXED enumerated candidate set:")
print(f"{'system':>18} " + " ".join(f"{x:>5}%" for x in NO))
for s in avail:
    print(f"{SYSN[s]:>18} " + " ".join(f"{v:6.2f}" for v in res[(s,'slic')]) + "   <- SLIC")
    print(f"{'':>18} " + " ".join(f"{v:6.2f}" for v in res[(s,'bic')]) + "   <- BIC")
