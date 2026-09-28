"""Supplementary Figs. S5-S8: Figure 3 perturbation study for false-positive rate, false-negative
rate, system-level recovery and log10 median SMAE. One script for all four so they stay identical
in layout; every axis gets ticks that span its data."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import h5py, numpy as np, os, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paperstyle import apply_paper_style, PCOLOR, PMARK
apply_paper_style(); matplotlib.rcParams['savefig.bbox']='standard'
def get(f,key):
    v=f[key][()]
    return f[v['parent_']][:] if hasattr(v,'dtype') and v.dtype.names and 'parent_' in v.dtype.names else v
def deref(f,ref):
    v=f[ref][()]
    return f[v['parent_']][:] if hasattr(v,'dtype') and v.dtype.names and 'parent_' in (v.dtype.names or ()) else v
ICS=['slic','aic','aicc','bic','hqic','kic','bc']; LBL=dict(zip(ICS,['SLIC','AIC','AICc','BIC','HQIC','KIC','BC']))
SYS=[('lorenz','Lorenz'),('rossler','Rossler'),('lotka-volterra','L-V'),('brusselator','Brusselator'),('van_der_pol','VdP'),('nonlin_pendulum','NLP')]
NOISE=[0,5,10,20,30,40]
NV={'lorenz':[5000,6000,7000,8000,9000],'rossler':[5000,6000,7000,8000,9000],'brusselator':[5000,6000,7000,8000,9000],
    'lotka-volterra':[500,1000,1500,2000,2500],'van_der_pol':[500,1000,1500,2000,2500],'nonlin_pendulum':[500,1000,1500,2000,2500]}
BLOCKS=[('dts','(a) Sampling frequency',[0,1,2,3],[r'$5\times dt_0$',r'$2\times dt_0$',r'$0.5\times dt_0$',r'$0.2\times dt_0$'],[s for s,_ in SYS]),
        ('Ts','(b) Trajectory length',[1,2,3,4],None,[s for s,_ in SYS]),
        ('ss','(c) Subsampling',[0,1,2],[r'$s=1$',r'$s=2$',r'$s=3$'],[s for s,_ in SYS]),
        ('ps','(d) ODE parameters',[0,1],['Param.\nset 1','Param.\nset 2'],['lorenz','lotka-volterra','brusselator','van_der_pol'])]
def metrics(A,Xt):
    XtT=np.asarray(Xt).T; tnz=np.abs(XtT)>1e-12; l1=np.abs(XtT).sum(); o={'fpr':[],'fnr':[],'rec':[],'smae':[]}
    for M in A:
        M=M if M.shape==XtT.shape else M.T; p=np.abs(M)>1e-12
        TP=np.sum(p&tnz);FP=np.sum(p&~tnz);TN=np.sum(~p&~tnz);FN=np.sum(~p&tnz)
        o['fpr'].append(FP/(FP+TN) if FP+TN else 0); o['fnr'].append(FN/(FN+TP) if FN+TP else 0)
        o['rec'].append(float(np.all(p==tnz))); o['smae'].append(np.abs(XtT-M).sum()/l1)
    return {'fpr':np.mean(o['fpr']),'fnr':np.mean(o['fnr']),'rec':np.mean(o['rec']),'smae':np.log10(np.median(o['smae']))}
CACHE=str(PLOTS/'fig3_metrics_cache.npy')
if os.path.exists(CACHE): V=np.load(CACHE,allow_pickle=True).item()
else:
    V={}
    for blk,_,levels,_,systems in BLOCKS:
        for s in systems:
            for ni,nl in enumerate(NOISE,1):
                with h5py.File(f'{RESULTS}/fig3_perturbations/{blk}/{s}_si{blk}_noise{ni}.jld','r') as f:
                    XT=get(f,'Xitrues')
                    for k in levels:
                        Xt=np.asarray(deref(f,XT[k]))
                        for ic in ICS: V[(blk,s,k,nl,ic)]=metrics(np.asarray(get(f,f'Xis_neffall_{ic}'))[k],Xt)
    np.save(CACHE,V,allow_pickle=True)
def cover_ticks(lo,hi,unit):
    """Ticks on multiples of `unit` whose outermost values enclose [lo, hi]."""
    a=np.floor(lo/unit+1e-9)*unit; b=np.ceil(hi/unit-1e-9)*unit
    if b<=a: b=a+unit
    return np.arange(a,b+unit/2,unit)
YL={'fpr':'False-positive rate','fnr':'False-negative rate','rec':'System-level recovery','smae':r'$\log_{10}$ SMAE (median)'}
def make(metric,out):
    fig=plt.figure(figsize=(29.7,20.6))
    L,R,T,B=0.035,0.985,0.925,0.075; gx,gy=0.07,0.075
    bw=(R-L-gx)/2; ph=(T-B-gy)/7                      # 4 rows on top, 3 below, all the same height
    origins=[(L,B+3*ph+gy,4),(L+bw+gx,B+3*ph+gy,4),(L,B,3),(L+bw+gx,B,3)]
    for (blk,title,levels,rowlab,systems),(ox,oy,slots) in zip(BLOCKS,origins):
        nr=len(levels); nc=6; pw=bw/nc; bh=slots*ph
        fig.text(ox-0.02,oy+bh+0.012,title,fontsize=24,ha='left',va='bottom')
        for ri,k in enumerate(levels):
            row_axes=[]
            for ci,s in enumerate(systems):
                ax=fig.add_axes([ox+ci*pw+0.18*pw, oy+bh-(ri+1)*ph+0.20*ph, 0.72*pw, 0.66*ph]); row_axes.append(ax)
                for ic in ['bc','kic','hqic','bic','aicc','aic','slic']:
                    y=[V[(blk,s,k,nl,ic)][metric] for nl in NOISE]
                    ax.plot(NOISE,y,marker=PMARK[ic],ms=8 if ic=='slic' else 6,color=PCOLOR[ic],lw=3.0 if ic=='slic' else 1.6,
                            markeredgecolor='black',markeredgewidth=0.4,zorder=6 if ic=='slic' else 3,label=LBL[ic])
                ax.set_xlim(-2,42); ax.set_xticks([0,20,40]); ax.tick_params(labelsize=15,length=5)
                if ri==nr-1: ax.set_xlabel('% Noise',fontsize=17)
                else: ax.set_xticklabels([])
                if ri==0: ax.set_title(dict(SYS)[s],fontsize=19,pad=18 if blk=='Ts' else 6)
                if blk=='Ts': ax.text(0.5,1.02,f'$N={NV[s][k]}$',transform=ax.transAxes,ha='center',va='bottom',fontsize=13)
            # one y range per row, with ticks enclosing every panel's data
            ys=[V[(blk,s,k,nl,ic)][metric] for s in systems for nl in NOISE for ic in ICS]
            if metric=='smae':
                tk=cover_ticks(min(ys),max(ys),1.0 if max(ys)-min(ys)<=3 else 2.0)
            else: tk=np.array([0,0.5,1.0])
            pad=0.06*(tk[-1]-tk[0])
            for ci,ax in enumerate(row_axes):
                ax.set_ylim(tk[0]-pad,tk[-1]+pad); ax.set_yticks(tk)
                if ci==0:
                    ax.set_yticklabels([f'{t:g}' for t in tk])
                    if rowlab: ax.set_ylabel(rowlab[ri],fontsize=17)
                else: ax.set_yticklabels([])
    h,l=fig.axes[0].get_legend_handles_labels(); order=[l.index(x) for x in ['SLIC','AIC','AICc','BIC','HQIC','KIC','BC']]
    fig.legend([h[i] for i in order],[l[i] for i in order],loc='lower center',ncol=7,fontsize=21,bbox_to_anchor=(0.5,0.0))
    fig.text(0.5,0.985,YL[metric],ha='center',va='top',fontsize=26)
    for e in ('png','pdf'): fig.savefig(f'{out}.{e}',dpi=250 if e=='png' else None)
    return fig
if __name__=='__main__':
    from tickcheck import check
    for m,name in [('fpr','SI_fig3_fpr'),('fnr','SI_fig3_fnr'),('rec','SI_fig3_rec'),('smae','SI_fig3_smae')]:
        fig=make(m,str(PLOTS/name)); check(fig,name)
        fig.canvas.draw(); r=fig.canvas.get_renderer()
        ov=lambda p,q: not (p.x1<=q.x0 or q.x1<=p.x0 or p.y1<=q.y0 or q.y1<=p.y0)
        it=[t.get_window_extent(renderer=r) for a in fig.axes for t in [a.title,a.xaxis.label,a.yaxis.label]+list(a.texts)+a.get_xticklabels()+a.get_yticklabels() if t.get_text().strip()]
        it+=[t.get_window_extent(renderer=r) for t in fig.texts]+[fig.legends[0].get_window_extent(renderer=r)]
        print(f"   text overlaps: {sum(1 for i in range(len(it)) for j in range(i+1,len(it)) if ov(it[i],it[j]))}")
        plt.close(fig)
