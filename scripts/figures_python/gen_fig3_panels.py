import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, PLOTS
_R, _P = str(RESULTS), str(PLOTS)
import h5py, numpy as np, glob, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from paperstyle import apply_paper_style, PCOLOR, PLW, PMS, PMARK, PORDER, PLABEL
from mlib import get, all_metrics
apply_paper_style()
def deref(f,ref):
    o=f[ref]; v=o[()]
    if hasattr(v,'dtype') and v.dtype.names and 'parent_' in (v.dtype.names or ()): return f[v['parent_']][:]
    return v
SYS=[('lorenz','Lorenz'),('rossler','Rossler'),('lotka-volterra','Lotka-Volterra'),
     ('brusselator','Brusselator'),('van_der_pol','Van der Pol'),('nonlin_pendulum','Nonlinear pendulum')]
NO=[0,5,10,20,30,40]

ROWLAB={'dts':[r'$5\times dt_0$',r'$2\times dt_0$',r'$0.5\times dt_0$',r'$0.2\times dt_0$'],
        'ss':[r'$s=1$',r'$s=2$',r'$s=3$'],
        'ps':['Param. set 1','Param. set 2'],
        'Ts':[r'$N_1$',r'$N_2$',r'$N_3$',r'$N_4$',r'$N_5$']}

COND={'dts':'(a) Sampling frequency','Ts':'(b) Trajectory length',
      'ss':'(c) Subsampling','ps':'(d) ODE parameters'}

def lvl(cond,sysf,ic,metric):
    labs=None;mat=None
    for ni in range(1,7):
        g=glob.glob(f'{_R}/fig3_perturbations/**/{cond}/{sysf}_si{cond}_noise{ni}.jld',recursive=True)
        if not g:
            if mat is not None: mat[:,ni-1]=np.nan
            continue
        with h5py.File(g[0],'r') as f:
            A=np.asarray(get(f,f'Xis_neffall_{ic}')); Xts=get(f,'Xitrues')
            if labs is None:
                labs=[(x.decode() if isinstance(x,bytes) else x).replace('Δ',r'$\Delta$') for x in get(f,'labels')]
                mat=np.full((A.shape[0],6),np.nan)
            for l in range(A.shape[0]):
                Xt=deref(f,Xts[l]) if not isinstance(Xts[l],np.ndarray) else Xts[l]
                mat[l,ni-1]=all_metrics(A[l],Xt)[metric]
    return labs,mat

def build(cond,metric,ylab,outfile,logy=False):
    avail=[(s,l) for s,l in SYS if glob.glob(f'{_R}/fig3_perturbations/**/{cond}/{s}_si{cond}_noise1.jld',recursive=True)]
    labs,_=lvl(cond,avail[0][0],'slic',metric)
    NRW,NCL=len(labs),len(avail)
    TS,LS,TK,LG=38,37,31,40
    FW,FH=6.0*NCL,5.0*NRW+2.2          # +2.2in reserved for suptitle & xlabel
    fig=plt.figure(figsize=(FW,FH))
    top_in,bot_in,left_in,right_in=1.75,1.05,2.05,4.6   # absolute margins
    gs=GridSpec(NRW,NCL,figure=fig,hspace=0.30,wspace=0.20,
                left=left_in/FW,right=1-right_in/FW,
                top=1-top_in/FH,bottom=bot_in/FH)
    first=None
    for cj,(sysf,slbl) in enumerate(avail):
        mats={ic:lvl(cond,sysf,ic,metric)[1] for ic in PORDER}
        L=lvl(cond,sysf,'slic',metric)[0]
        for ri in range(NRW):
            ax=fig.add_subplot(gs[ri,cj])
            if first is None: first=ax
            for ic in PORDER:
                y=mats[ic][ri]
                if logy: y=np.log10(np.clip(y,1e-12,None))
                ax.plot(NO,y,marker=PMARK[ic],ms=PMS[ic]-2,color=PCOLOR[ic],lw=PLW[ic]-0.5,
                        label=PLABEL[ic],zorder=6 if ic=='slic' else 2,
                        markeredgecolor='black',markeredgewidth=0.7)
            ax.set_xticks(NO); ax.tick_params(labelsize=TK,pad=7)
            if not logy: ax.set_ylim(-0.05,1.07); ax.set_yticks([0,0.5,1.0])
            if ri==0: ax.set_title(slbl,fontsize=TS,pad=16)
            if cj>0: ax.set_yticklabels([])
            else:
                rl=ROWLAB.get(cond)
                ax.set_ylabel(f'{rl[ri] if rl and ri<len(rl) else L[ri]}\n{ylab}',
                              fontsize=LS-6,labelpad=14)
            if ri<NRW-1: ax.set_xticklabels([])
            else: ax.set_xlabel('Noise level (%)',fontsize=LS,labelpad=12)
    h,l=first.get_legend_handles_labels()
    fig.legend(h,l,loc='center left',bbox_to_anchor=(1-(right_in-0.55)/FW,0.5),fontsize=LG,
               handlelength=0.85,handletextpad=0.28,labelspacing=0.22,borderpad=0.30)
    fig.suptitle(COND[cond],fontsize=TS+6,fontweight='bold',y=1-0.30/FH,va='top')
    fig.savefig(outfile)
    for _e in ('pdf','svg'): fig.savefig(outfile.rsplit('.',1)[0]+'.'+_e)
    plt.close(fig); print("wrote",outfile.split('/')[-1])

if __name__=='__main__':
    cond=sys.argv[1] if len(sys.argv)>1 else 'dts'
    build(cond,'acc','Accuracy',f'{_P}/fig3_{cond}.png')
