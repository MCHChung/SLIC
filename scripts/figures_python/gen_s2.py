import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, PLOTS
_R, _P = str(RESULTS), str(PLOTS)
import h5py, numpy as np, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from paperstyle import apply_paper_style, PCOLOR, PLW, PMS, PMARK, PORDER, PLABEL
from mlib import get, all_metrics
apply_paper_style()
SYS=[('lorenz','Lorenz'),('rossler','Rossler'),('lotka-volterra','Lotka-Volterra'),
     ('brusselator','Brusselator'),('van_der_pol','Van der Pol'),('nonlin_pendulum','Nonlinear pendulum')]
NO=[0,5,10,20,30,40]
D={}
for s,_ in SYS:
    for ni in range(1,7):
        p=glob.glob(f'{_R}/fig2_benchmarks/**/{s}_neffall_noise{ni}_results.jld',recursive=True)[0]
        with h5py.File(p,'r') as f:
            Xt=get(f,'Xitrue')
            for ic in PORDER: D[(s,ni,ic)]=all_metrics(get(f,f'Xis_neffall_{ic}'),Xt)
ser=lambda s,ic,m:[D[(s,ni,ic)][m] for ni in range(1,7)]
def summ(metric):
    mu=np.array([[np.mean([D[(s,ni,ic)][metric] for s,_ in SYS]) for ni in range(1,7)] for ic in PORDER])
    se=np.array([[np.std([D[(s,ni,ic)][metric] for s,_ in SYS])/np.sqrt(6) for ni in range(1,7)] for ic in PORDER])
    return mu,se

def build(ROWS, outfile, figw=41, rowh=5.6):
    TS,LS,TK,LG=38,37,31,40
    NR=len(ROWS)
    fig=plt.figure(figsize=(figw,rowh*NR))
    outer=GridSpec(NR,1,figure=fig,hspace=0.38,
                   left=0.055,right=0.905,top=0.955,bottom=0.065)
    first=None
    for ri,(letter,metric,ylab,logy) in enumerate(ROWS):
        # uniform column spacing keeps the x-grid aligned across rows and
        # leaves room for the SMAE row's per-panel tick labels
        ws=0.20
        inner=GridSpecFromSubplotSpec(1,7,subplot_spec=outer[ri],wspace=ws)
        row=[]
        for cj,(sysf,lbl) in enumerate(SYS):
            ax=fig.add_subplot(inner[0,cj]); row.append(ax)
            if first is None: first=ax
            drawn=[]
            for ic in PORDER:
                y=np.array(ser(sysf,ic,metric),dtype=float)
                if logy: y=np.log10(np.clip(y,1e-12,None))
                # criteria frequently select IDENTICAL models (e.g. BIC==KIC on
                # Lorenz). Nudge exact/near duplicates apart along x so the
                # coincidence is visible rather than hidden under one curve.
                xs=np.array(NO,dtype=float)
                if logy:  # only the SMAE row, where genuine coincidences hide curves
                    dup=sum(1 for yy in drawn if np.allclose(yy,y,rtol=0,atol=1e-9))
                    if dup: xs=xs+dup*0.55
                drawn.append(y)
                ax.plot(xs,y,marker=PMARK[ic],ms=PMS[ic]-2,color=PCOLOR[ic],
                        lw=PLW[ic]-0.5,label=PLABEL[ic],zorder=6 if ic=='slic' else 2,
                        markeredgecolor='black',markeredgewidth=0.7,
                        alpha=1.0 if ic=='slic' else 0.85)
            if ri==0: ax.set_title(lbl,fontsize=TS,pad=16)
            if logy:
                v=np.log10(np.array([ser(sysf,ic,metric) for ic in PORDER]).ravel())
                v=v[np.isfinite(v)]
                ax.set_ylim(v.min()-0.75,v.max()+0.75)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
        axs=fig.add_subplot(inner[0,6]); row.append(axs)
        mu,se=summ(metric)
        for m,ic in enumerate(PORDER):
            y,lo,hi=mu[m],mu[m]-se[m],mu[m]+se[m]
            if logy:
                y=np.log10(np.clip(y,1e-12,None))
                lo=np.log10(np.clip(mu[m]-se[m],1e-12,None)); hi=np.log10(np.clip(mu[m]+se[m],1e-12,None))
            axs.plot(NO,y,marker=PMARK[ic],ms=PMS[ic]-2,color=PCOLOR[ic],lw=PLW[ic]-0.5,
                     zorder=6 if ic=='slic' else 2,markeredgecolor='black',markeredgewidth=0.7)
            axs.fill_between(NO,lo,hi,color=PCOLOR[ic],alpha=.12,zorder=1)
        if ri==0: axs.set_title('All systems\n(mean)',fontsize=TS,pad=16,fontweight='bold')
        if logy:
            v=np.log10(mu[mu>0]); axs.set_ylim(v.min()-0.75,v.max()+0.75)
            axs.yaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
        for cj,ax in enumerate(row):
            ax.set_xticks(NO); ax.tick_params(labelsize=TK,pad=7)
            if not logy:
                ax.set_ylim(-0.05,1.07); ax.set_yticks([0,0.5,1.0])
                if cj>0: ax.set_yticklabels([])
            if ri<NR-1: ax.set_xticklabels([])
            else: ax.set_xlabel('Noise level (%)',fontsize=LS,labelpad=12)
        row[0].set_ylabel(ylab,fontsize=LS,labelpad=16)
        row[0].text(-0.42,1.07,f'({letter})',transform=row[0].transAxes,fontsize=48,
                    fontweight='bold',va='bottom',ha='right')
    h,l=first.get_legend_handles_labels()
    fig.legend(h,l,loc='center left',bbox_to_anchor=(0.912,0.5),fontsize=LG,
               handlelength=0.85,handletextpad=0.28,labelspacing=0.22,borderpad=0.30)
    fig.savefig(outfile)
    for _ext in ('pdf','svg'):  # VECTOR_EXPORT
        fig.savefig(outfile.rsplit('.',1)[0]+'.'+_ext)
    plt.close(fig)
    print("wrote",outfile.split('/')[-1])

# MAIN: exactly the metrics R2 asked for
# SI: the additional metrics R2 named but did not request
build([('a','f1','F1 score',False),
       ('b','bacc','Balanced\naccuracy',False),
       ('c','prec','Precision',False)],
      f'{_P}/SI_extra_metrics.png')
